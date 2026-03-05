import json
import os
import sys
import uuid
import asyncio
import logging
from typing import Dict, Any, Optional
from deepgram import LiveOptions

from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from loguru import logger
import uvicorn

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s")

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.frames.frames import LLMMessagesFrame, AudioRawFrame, EndFrame, ErrorFrame
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

logger.remove()
logger.add(sys.stderr, level="DEBUG")

app = FastAPI()

active_call_configs: Dict[str, Any] = {}

FS_HOST = os.getenv("FREESWITCH_HOST", "10.0.1.7")
FS_ESL_PORT = int(os.getenv("FREESWITCH_ESL_PORT", "8021"))
FS_ESL_PASSWORD = os.getenv("FREESWITCH_ESL_PASSWORD", "ClueCon")
PIPECAT_WS_BASE = os.getenv("PIPECAT_WS_BASE_URL", "ws://10.0.1.7:8000")


class OutboundCallRequest(BaseModel):
    phone_number: str
    caller_id: str = "2167064380"
    system_prompt: Optional[str] = "Sen yardımsever bir asistansın."

    # STT
    stt_provider: Optional[str] = "deepgram"
    stt_model: Optional[str] = "nova-3"          # nova-3 destekli model adı
    stt_language: Optional[str] = "tr"
    stt_sample_rate: Optional[int] = 8000

    # LLM
    llm_provider: Optional[str] = "openai"
    llm_model: Optional[str] = "gpt-4o"

    # TTS
    tts_provider: Optional[str] = "cartesia"
    tts_voice_id: Optional[str] = "39f753ef-b0eb-41cd-aa53-2f3c284f948f"


async def esl_originate(originate_cmd: str) -> str:
    reader, writer = await asyncio.open_connection(FS_HOST, FS_ESL_PORT)
    try:
        await reader.readuntil(b"auth/request\n\n")
        writer.write(f"auth {FS_ESL_PASSWORD}\n\n".encode())
        await writer.drain()
        resp = await reader.readuntil(b"\n\n")
        if b"+OK accepted" not in resp:
            raise Exception(f"ESL auth başarısız: {resp}")
        cmd = f"bgapi originate {originate_cmd}\n\n"
        writer.write(cmd.encode())
        await writer.drain()
        response = await reader.readuntil(b"\n\n")
        return response.decode(errors="replace")
    finally:
        writer.close()
        await writer.wait_closed()


def service_factory(config: dict):
    stt_provider = config.get("stt_provider", "deepgram")
    stt_model = config.get("stt_model", "nova-3")
    stt_language = config.get("stt_language", "tr")
    stt_sample_rate = config.get("stt_sample_rate", 8000)

    if stt_provider == "deepgram":
        # nova-3 ile çalışan minimal ve güvenli parametreler
        # utterance_end_ms ve endpointing nova-3 ile 400 hatasına yol açabiliyor
        # bu yüzden sadece zorunlu parametreleri gönderiyoruz
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            live_options=LiveOptions{
                "model": stt_model,
                "language": stt_language,
                "encoding": "linear16",
                "sample_rate": stt_sample_rate,
                "channels": 1,
                "interim_results": True,
                "punctuate": True,
                "smart_format": False,   # smart_format kapalı - nova-3 ile 400 riski var
                "vad_events": False,
                "profanity_filter": False,
            }
        )
    else:
        raise ValueError(f"Desteklenmeyen STT provider: {stt_provider}")

    system_prompt = config.get("system_prompt", "Sen yardımsever bir asistansın.")
    context = LLMContext(messages=[{"role": "system", "content": system_prompt}])

    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(stop_secs=0.3, start_secs=0.15)
    )

    context_aggregator_pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=vad_analyzer),
    )

    llm_provider = config.get("llm_provider", "openai")
    if llm_provider == "openai":
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=config.get("llm_model", "gpt-4o")
        )
    elif llm_provider == "anthropic":
        llm = AnthropicLLMService(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=config.get("llm_model", "claude-3-5-sonnet-20240620"),
            system=system_prompt
        )
    else:
        raise ValueError(f"Desteklenmeyen LLM provider: {llm_provider}")

    tts_provider = config.get("tts_provider", "cartesia")
    if tts_provider == "cartesia":
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id=config.get("tts_voice_id") or "39f753ef-b0eb-41cd-aa53-2f3c284f948f"
        )
    elif tts_provider == "elevenlabs":
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=config.get("tts_voice_id") or "21m00Tcm4TlvDq8ikWAM"
        )
    else:
        raise ValueError(f"Desteklenmeyen TTS provider: {tts_provider}")

    return stt, llm, tts, context_aggregator_pair


class WebSocketOutput(FrameProcessor):
    def __init__(self, websocket: WebSocket):
        super().__init__()
        self.websocket = websocket

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame):
            try:
                await self.websocket.send_bytes(frame.audio)
            except Exception as e:
                logger.error(f"WebSocket ses gönderme hatası: {e}")


async def websocket_input(websocket: WebSocket):
    try:
        while True:
            message = await websocket.receive()
            if 'text' in message and message['text']:
                logger.info(f"FreeSWITCH Metadata: {message['text']}")
                continue
            elif 'bytes' in message and message['bytes']:
                logger.info(f"Binary ses paketi geldi! Boyut: {len(message['bytes'])} bytes")
                yield AudioRawFrame(audio=message['bytes'], sample_rate=8000, num_channels=1)
            elif message.get('type') == 'websocket.disconnect':
                logger.info("WebSocket disconnect yakalandı.")
                break
    except Exception as e:
        logger.error(f"WebSocket okuma hatası: {e}")
    yield EndFrame()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket bağlantısı kuruldu.")
    logger.info(f"📡 WS Bağlantısı - Query Params: {dict(websocket.query_params)}")
    logger.info(f"📡 WS Bağlantısı - Headers: {dict(websocket.headers)}")
    logger.info(f"📡 WS Bağlantısı - call_id alındı: {websocket.query_params.get('call_id')}")
    call_id = websocket.query_params.get("call_id")
    if not call_id or call_id not in active_call_configs:
        logger.error(f"Geçersiz veya eksik call_id: {call_id}. Bağlantı reddediliyor.")
        await websocket.close(code=1008, reason="Geçersiz veya eksik call_id")
        return

    config = active_call_configs[call_id]
    logger.info(f"Config başarıyla bulundu. call_id={call_id}")

    stt, llm, tts, context_aggregator_pair = service_factory(config)
    output_processor = WebSocketOutput(websocket)

    pipeline = Pipeline([
        stt,
        context_aggregator_pair.user(),
        llm,
        tts,
        output_processor,
        context_aggregator_pair.assistant()
    ])

    runner = PipelineRunner()
    task = PipelineTask(pipeline)

    async def push_audio():
        try:
            async for frame in websocket_input(websocket):
                if isinstance(frame, AudioRawFrame):
                    logger.info(f"Audio frame alındı! Uzunluk: {len(frame.audio)} bytes, "
                                f"sample_rate: {frame.sample_rate}, channels: {frame.num_channels}")
                else:
                    logger.debug(f"Non-audio frame: {type(frame).__name__}")
                
                await task.queue_frame(frame)
        except WebSocketDisconnect:
            logger.info("WebSocket istemci tarafından kapatıldı (normal çıkış).")
            await task.queue_frame(EndFrame())
        except Exception as e:
            logger.error(f"Ses push hatası: {e}")
            await task.queue_frame(ErrorFrame(f"Audio input error: {e}"))
        # finally yok – runner doğal olarak bitsin, zorla iptal etme

    logger.info("Pipeline başlatılıyor...")
    try:
        await asyncio.gather(runner.run(task), push_audio())
    except Exception as e:
        logger.error(f"Pipeline genel hatası: {e}")
    finally:
        # Sadece temizlik: config sil ve websocket kapat
        if call_id in active_call_configs:
            del active_call_configs[call_id]
            logger.info(f"Config temizlendi. call_id={call_id}")
        try:
            await websocket.close()
        except:
            pass

@app.get("/")
async def root():
    return {"status": "OK"}


@app.get("/health")
async def health():
    return {"status": "healthy", "active_calls": len(active_call_configs)}


@app.post("/outbound-call")
async def start_outbound_call(request: OutboundCallRequest, background_tasks: BackgroundTasks):
    call_id = str(uuid.uuid4())
    active_call_configs[call_id] = request.dict()

    ws_url = f"{PIPECAT_WS_BASE}/ws?call_id={call_id}"

    originate_cmd = (
        f"{{"
        f"origination_uuid={call_id},"
        f"origination_caller_id_number={request.caller_id},"
        f"origination_caller_id_name=AI_Asistan,"
        f"pipecat_call_id={call_id},"
        f"pipecat_ws_url=ws://pipecatcon:8000/ws?call_id={call_id},"
        f"ignore_early_media=true,"
        f"progress_timeout=60,"
        f"absolute_codec_string=PCMU,"   
        f"}}sofia/gateway/netgsm/{request.phone_number} "
        f"&lua(pipecat_connect.lua)"
    )
           
    try:
        result = await esl_originate(originate_cmd)
        logger.info(f"ESL cevabı: {result}")
        return {"status": "initiated", "call_id": call_id, "esl_response": result}
    except Exception as e:
        logger.error(f"ESL hatası: {e}")
        return {"status": "esl_error", "error": str(e), "call_id": call_id}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
