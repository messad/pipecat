import os
import sys
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, BackgroundTasks, HTTPException
from pydantic import BaseModel
from loguru import logger
import uvicorn

# Pipecat importları
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.aggregators.llm_response import LLMUserResponseAggregator
from pipecat.services.openai import OpenAILLMService
from pipecat.services.anthropic import AnthropicLLMService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.transports.network.websocket_server import WebsocketServerTransport, WebsocketServerParams
from pipecat.frames.frames import LLMMessagesFrame

# ESL (FreeSWITCH Event Socket) - outbound arama için
import ESL

# Loglama
logger.remove()
logger.add(sys.stderr, level="DEBUG")

app = FastAPI()

# Aktif çağrı konfigürasyonları (hafızada)
active_call_configs: Dict[str, Any] = {}

# Ortam değişkenlerinden FreeSWITCH ESL bağlantı bilgileri
FS_HOST = os.getenv("FREESWITCH_HOST", "127.0.0.1")
FS_ESL_PORT = int(os.getenv("FREESWITCH_ESL_PORT", "8021"))
FS_ESL_PASSWORD = os.getenv("FREESWITCH_ESL_PASSWORD", "ClueCon")

# Pipecat server'ın dışarıya açık adresi (FreeSWITCH buraya WS bağlayacak)
PIPECAT_WS_BASE = os.getenv("PIPECAT_WS_BASE_URL", "ws://localhost:8000")


class OutboundCallRequest(BaseModel):
    phone_number: str
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    stt_provider: str = "deepgram"
    tts_provider: str = "cartesia"
    tts_voice_id: Optional[str] = None          # Boşsa provider default'u kullanır
    system_prompt: str = "Sen yardımsever bir asistansın."
    caller_id: str = "2167064380"               # NetGSM'deki numaranız


# --- DİNAMİK SERVİS FABRİKASI ---
def service_factory(config: dict):
    # 1. STT
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        language="tr"  # Türkçe - gerekirse config'den alınabilir
    )

    # 2. LLM - system_prompt context olarak iletiliyor
    system_messages = [{"role": "system", "content": config.get("system_prompt", "Sen yardımsever bir asistansın.")}]

    if config.get("llm_provider") == "anthropic":
        llm = AnthropicLLMService(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=config.get("llm_model", "claude-3-5-sonnet-20240620"),
            system=config.get("system_prompt", "Sen yardımsever bir asistansın.")
        )
    else:
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=config.get("llm_model", "gpt-4o")
        )

    # 3. TTS
    if config.get("tts_provider") == "elevenlabs":
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=config.get("tts_voice_id") or "21m00Tcm4TlvDq8ikWAM"
        )
    else:
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id=config.get("tts_voice_id") or "a0e99841-438c-4a64-b679-ae501e7d6091"
        )

    return stt, llm, tts, system_messages


# --- OUTBOUND ARAMA: n8n bu endpoint'i çağırır ---
@app.post("/outbound-call")
async def start_outbound_call(request: OutboundCallRequest, background_tasks: BackgroundTasks):
    call_id = str(uuid.uuid4())
    active_call_configs[call_id] = request.dict()

    ws_path = f"/ws/{call_id}"
    ws_url = f"{PIPECAT_WS_BASE}{ws_path}"

    logger.info(f"Outbound arama başlatılıyor. call_id={call_id}, hedef={request.phone_number}")

    # FreeSWITCH ESL üzerinden originate komutu gönder
    try:
        con = ESL.ESLconnection(FS_HOST, str(FS_ESL_PORT), FS_ESL_PASSWORD)
        if not con.connected():
            raise Exception("FreeSWITCH ESL bağlantısı kurulamadı")

        # originate komutu:
        # - call_id ve ws_url'i channel variable olarak geçiyoruz
        # - Dialplan'da bu variable'ları kullanarak audio_stream başlatacağız
        originate_cmd = (
            f"originate {{"
            f"origination_caller_id_number={request.caller_id},"
            f"origination_caller_id_name={request.caller_id},"
            f"pipecat_call_id={call_id},"
            f"pipecat_ws_url={ws_url},"
            f"ignore_early_media=false,"
            f"progress_timeout=60"
            f"}}sofia/gateway/netgsm/{request.phone_number} "
            f"&lua(/usr/share/freeswitch/scripts/pipecat_connect.lua)"
        )

        logger.debug(f"ESL originate: {originate_cmd}")
        result = con.api("bgapi", originate_cmd)
        logger.info(f"ESL cevabı: {result.getBody() if result else 'None'}")
        con.disconnect()

    except Exception as e:
        logger.error(f"ESL hatası: {e}")
        # ESL bağlanamasa bile call_id'yi dön, manual test için
        return {
            "status": "esl_error",
            "error": str(e),
            "call_id": call_id,
            "websocket_url": ws_url,
            "note": "FreeSWITCH'e bağlanılamadı, manual originate gerekebilir"
        }

    return {
        "status": "initiated",
        "call_id": call_id,
        "websocket_url": ws_url,
        "phone_number": request.phone_number
    }


# --- INBOUND / MANUAL TEST endpoint'i ---
# FreeSWITCH sabit bir path ile bağlanır, system_prompt default gelir
@app.post("/inbound-call")
async def register_inbound_call(background_tasks: BackgroundTasks):
    call_id = str(uuid.uuid4())
    active_call_configs[call_id] = {
        "llm_provider": "openai",
        "llm_model": "gpt-4o",
        "stt_provider": "deepgram",
        "tts_provider": "cartesia",
        "system_prompt": "Sen yardımsever bir asistansın.",
    }
    ws_url = f"{PIPECAT_WS_BASE}/ws/{call_id}"
    logger.info(f"Inbound çağrı kaydedildi. call_id={call_id}")
    return {"call_id": call_id, "websocket_url": ws_url}


# --- WEBSOCKET ENDPOINT: FreeSWITCH buraya bağlanır ---
@app.websocket("/ws/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str):
    await websocket.accept()
    logger.info(f"FreeSWITCH WebSocket bağlandı. call_id={call_id}")

    config = active_call_configs.get(call_id)
    if not config:
        logger.warning(f"call_id bulunamadı: {call_id}, default config kullanılıyor")
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
            "system_prompt": "Sen yardımsever bir asistansın.",
            "tts_provider": "cartesia"
        }

    # FreeSWITCH → mod_audio_stream → 8000Hz 16bit mono L16
    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            add_wav_header=False
        )
    )

    stt, llm, tts, system_messages = service_factory(config)

    pipeline = Pipeline([
        transport.input(),
        stt,
        LLMUserResponseAggregator(),
        llm,
        tts,
        transport.output()
    ])

    runner = PipelineRunner()
    task = PipelineTask(pipeline)

    # OpenAI için system_prompt'u initial message olarak gönder
    if config.get("llm_provider") != "anthropic":
        await task.queue_frame(LLMMessagesFrame(system_messages))

    await runner.run(task)

    # Temizlik
    if call_id in active_call_configs:
        del active_call_configs[call_id]
        logger.info(f"Çağrı tamamlandı, config temizlendi. call_id={call_id}")


# --- SAĞLIK KONTROLÜ ---
@app.get("/health")
async def health():
    return {"status": "ok", "active_calls": len(active_call_configs)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
