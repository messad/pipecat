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

# Pipecat importları (Deprecation uyarıları giderildi, raw audio için frame eklendi)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.aggregators.llm_response import LLMUserResponseAggregator
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.openai import OpenAILLMService
from pipecat.services.anthropic import AnthropicLLMService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.transports.websocket.server import WebsocketServerTransport, WebsocketServerParams
from pipecat.frames.frames import LLMMessagesFrame, AudioRawFrame, EndFrame

logger.remove()
logger.add(sys.stderr, level="DEBUG")

app = FastAPI()

active_call_configs: Dict[str, Any] = {}

FS_HOST = os.getenv("FREESWITCH_HOST", "freeswitchcon")
FS_ESL_PORT = int(os.getenv("FREESWITCH_ESL_PORT", "8021"))
FS_ESL_PASSWORD = os.getenv("FREESWITCH_ESL_PASSWORD", "ClueCon")
PIPECAT_WS_BASE = os.getenv("PIPECAT_WS_BASE_URL", "pipecatcon:9001")


class OutboundCallRequest(BaseModel):
    phone_number: str
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    stt_provider: str = "deepgram"
    tts_provider: str = "cartesia"
    tts_voice_id: Optional[str] = None
    system_prompt: str = "Sen yardımsever bir asistansın."
    caller_id: str = "2167064380"


# DOKUNULMADI - Senin orijinal kodun
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


# DOKUNULMADI - Senin orijinal kodun
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

# DOKUNULMADI - Senin orijinal kodun
@app.post("/outbound-call")
async def start_outbound_call(request: OutboundCallRequest, background_tasks: BackgroundTasks):
    call_id = str(uuid.uuid4())
    active_call_configs[call_id] = request.dict()

    raw_url = PIPECAT_WS_BASE
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
        f"&socket({raw_url} async full)"
    )

    try:
        result = await esl_originate(originate_cmd)
        logger.info(f"ESL cevabı: {result}")
        return {"status": "initiated", "call_id": call_id, "raw_url": raw_url, "esl_response": result}
    except Exception as e:
        logger.error(f"ESL hatası: {e}")
        return {
            "status": "esl_error",
            "error": str(e),
            "call_id": call_id,
            "raw_url": raw_url,
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
    raw_url = PIPECAT_WS_BASE
    return {"call_id": call_id, "raw_url": raw_url}


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


# --- PİPECAT 0.0.103 RAW TCP EKLENTİLERİ ---

class RawTCPOutput(FrameProcessor):
    """Pipeline'dan gelen TTS sesini FreeSWITCH'e gönderir"""
    def __init__(self, writer):
        super().__init__()
        self.writer = writer

    async def process_frame(self, frame, direction):
        if isinstance(frame, AudioRawFrame):
            self.writer.write(frame.audio)
            await self.writer.drain()
        elif isinstance(frame, EndFrame):
            self.writer.close()
        await self.push_frame(frame, direction)


async def raw_tcp_input(reader):
    """FreeSWITCH'ten gelen sesi okuyup Pipeline'a verir"""
    try:
        while True:
            data = await reader.read(320)  # 20ms L16
            if not data:
                yield EndFrame()
                break
            yield AudioRawFrame(audio=data, sample_rate=8000, num_channels=1)
    except Exception as e:
        logger.error(f"Input hata: {e}")
        yield EndFrame()

# --- SENİN YAZDIĞIN TCP LİSTENER ---
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


# --- GÜNCELLENEN TEK FONKSİYON: Echo yerine Pipeline tetikliyor ---
async def handle_raw_client(client, addr):
    try:
        reader, writer = await asyncio.open_connection(sock=client)
        
        # 1. FreeSWITCH'in gönderdiği detayları oku
        header = await reader.readuntil(b"\n\n")
        logger.info("FreeSWITCH bağlandı, header alındı.")

        # 2. KRİTİK NOKTA: FreeSWITCH'e "Bağlantıyı kabul ettim, kanalı aç" de
        writer.write(b"connect\n\n")
        await writer.drain()
        logger.info("FreeSWITCH'e 'connect' emri gönderildi, çağrı kilitlendi.")

        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
            "system_prompt": "Sen yardımsever bir asistansın.",
            "tts_provider": "cartesia"
        }
        stt, llm, tts, system_messages = service_factory(config)
        output_processor = RawTCPOutput(writer)

        pipeline = Pipeline([
            raw_tcp_input(reader),
            stt,
            LLMUserResponseAggregator(),
            llm,
            tts,
            output_processor
        ])

        runner = PipelineRunner()
        task = PipelineTask(pipeline)

        if config.get("llm_provider") != "anthropic":
            await task.queue_frame(LLMMessagesFrame(system_messages))

        await runner.run(task)

    except Exception as e:
        logger.error(f"Raw socket hata ({addr}): {e}")
    finally:
        client.close()


# Uygulama başlatıldığında raw listener'ı çalıştır
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(raw_socket_listener())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
