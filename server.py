import os
import sys
import uuid
import asyncio
import socket
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, BackgroundTasks
from pydantic import BaseModel
from loguru import logger
import uvicorn

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s")

# Pipecat importları
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.aggregators.llm_response import LLMUserResponseAggregator
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.websocket.server import WebsocketServerTransport, WebsocketServerParams
from pipecat.frames.frames import LLMMessagesFrame

logger.remove()
logger.add(sys.stderr, level="DEBUG")

app = FastAPI()

active_call_configs: Dict[str, Any] = {}

FS_HOST = os.getenv("FREESWITCH_HOST", "freeswitchcon")
FS_ESL_PORT = int(os.getenv("FREESWITCH_ESL_PORT", "8021"))
FS_ESL_PASSWORD = os.getenv("FREESWITCH_ESL_PASSWORD", "ClueCon")
PIPECAT_WS_BASE = os.getenv("PIPECAT_WS_BASE_URL", "ws://pipecatcon:8000")


class OutboundCallRequest(BaseModel):
    phone_number: str
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    stt_provider: str = "deepgram"
    tts_provider: str = "cartesia"
    tts_voice_id: Optional[str] = None
    system_prompt: str = "Sen yardımsever bir asistansın."
    caller_id: str = "2167064380"


async def esl_originate(originate_cmd: str) -> str:
    loop = asyncio.get_event_loop()

    def _send():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect((FS_HOST, FS_ESL_PORT))

        def recv_until(marker: bytes, max_bytes: int = 65536) -> bytes:
            buf = b""
            while marker not in buf:
                chunk = s.recv(4096)
                if not chunk:
                    raise Exception("ESL bağlantısı beklenmedik şekilde kapandı")
                buf += chunk
                if len(buf) > max_bytes:
                    raise Exception(f"ESL cevabı max_bytes ({max_bytes}) aştı, sonsuz döngü önlendi")
            return buf

        recv_until(b"auth/request")
        s.sendall(f"auth {FS_ESL_PASSWORD}\n\n".encode())
        resp = recv_until(b"\n\n")
        if b"+OK accepted" not in resp:
            raise Exception(f"ESL auth başarısız: {resp}")

        cmd = f"bgapi originate {originate_cmd}\n\n"
        s.sendall(cmd.encode())
        resp = recv_until(b"\n\n")
        s.close()
        return resp.decode(errors="replace")

    result = await loop.run_in_executor(None, _send)
    return result


def service_factory(config: dict):
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        language="tr"
    )

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

    ws_url = f"{PIPECAT_WS_BASE}/ws/{call_id}"
    logger.info(f"Outbound arama: call_id={call_id}, hedef={request.phone_number}")

    originate_cmd = (
        f"{{"
        f"origination_caller_id_number={request.caller_id},"
        f"origination_caller_id_name={request.caller_id},"
        f"pipecat_call_id={call_id},"
        f"ignore_early_media=true,"
        f"progress_timeout=60,"
        f"absolute_codec_string=PCMA"
        f"}}sofia/gateway/netgsm/{request.phone_number} "
        f"&socket({ws_url} async full)"
    )

    try:
        result = await esl_originate(originate_cmd)
        logger.info(f"ESL cevabı: {result}")
        return {"status": "initiated", "call_id": call_id, "websocket_url": ws_url, "esl_response": result}
    except Exception as e:
        logger.error(f"ESL hatası: {e}")
        return {
            "status": "esl_error",
            "error": str(e),
            "call_id": call_id,
            "websocket_url": ws_url,
            "note": "FreeSWITCH'e bağlanılamadı"
        }


@app.post("/inbound-call")
async def register_inbound_call():
    call_id = str(uuid.uuid4())
    active_call_configs[call_id] = {
        "llm_provider": "openai",
        "llm_model": "gpt-4o",
        "tts_provider": "cartesia",
        "system_prompt": "Sen yardımsever bir asistansın.",
    }
    ws_url = f"{PIPECAT_WS_BASE}/ws/{call_id}"
    return {"call_id": call_id, "websocket_url": ws_url}


@app.websocket("/ws/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str):
    await websocket.accept()
    logger.info(f"FreeSWITCH bağlandı. call_id={call_id}")

    config = active_call_configs.get(call_id) or {
        "llm_provider": "openai",
        "llm_model": "gpt-4o",
        "system_prompt": "Sen yardımsever bir asistansın.",
        "tts_provider": "cartesia"
    }

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

    if config.get("llm_provider") != "anthropic":
        await task.queue_frame(LLMMessagesFrame(system_messages))

    await runner.run(task)

    if call_id in active_call_configs:
        del active_call_configs[call_id]
        logger.info(f"Çağrı tamamlandı. call_id={call_id}")

async def raw_socket_listener():
    HOST = '0.0.0.0'
    PORT = 9001  # 8000'i bozmamak için ayrı port
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)
    logger.info(f"Raw TCP socket dinleniyor: {HOST}:{PORT}")

    loop = asyncio.get_running_loop()
    while True:
        client, addr = await loop.sock_accept(server)
        logger.info(f"FreeSWITCH raw bağlantı geldi: {addr}")
        asyncio.create_task(handle_raw_client(client, addr))

async def handle_raw_client(client, addr):
    try:
        # FreeSWITCH socket app'i text tabanlı basit protokol bekliyor
        data = await asyncio.get_running_loop().sock_recv(client, 4096)
        logger.info(f"Raw gelen veri ({addr}): {data.decode(errors='ignore')}")

        # Şimdilik basit echo testi (sonra pipeline tetikleme eklenecek)
        await asyncio.get_running_loop().sock_sendall(client, b"+OK connected to Pipecat\n")

        # Burada pipeline'ı başlatabilirsin (call_id parse et, config çek vs.)
        # Örnek: await start_pipeline_from_raw(call_id_from_data)
    except Exception as e:
        logger.error(f"Raw socket hata ({addr}): {e}")
    finally:
        client.close()

# Uygulama başlatıldığında raw listener'ı çalıştır
asyncio.create_task(raw_socket_listener())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
