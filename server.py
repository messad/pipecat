import os
import sys
import uuid
import asyncio
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks, WebSocket
from pydantic import BaseModel
from loguru import logger
import uvicorn

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s")

# Pipecat 0.0.103 Uyumlu Importlar
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
from pipecat.frames.frames import LLMMessagesFrame, AudioRawFrame, EndFrame

logger.remove()
logger.add(sys.stderr, level="DEBUG")

app = FastAPI()

active_call_configs: Dict[str, Any] = {}

FS_HOST = os.getenv("FREESWITCH_HOST", "freeswitchcon")
FS_ESL_PORT = int(os.getenv("FREESWITCH_ESL_PORT", "8021"))
FS_ESL_PASSWORD = os.getenv("FREESWITCH_ESL_PASSWORD", "ClueCon")

class OutboundCallRequest(BaseModel):
    phone_number: str
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    stt_provider: str = "deepgram"
    tts_provider: str = "cartesia"
    tts_voice_id: Optional[str] = None
    system_prompt: str = "Sen yardımsever bir asistansın."
    caller_id: str = "2167064380"

# --- SENİN ESL BAĞLANTI KODUN (DOKUNULMADI) ---
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

# --- SENİN SERVİS FABRİKAN (DOKUNULMADI) ---
def service_factory(config: dict):
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"), language="tr")
    system_messages = [{"role": "system", "content": config.get("system_prompt", "Sen yardımsever bir asistansın.")}]
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model=config.get("llm_model", "gpt-4o"))
    tts = CartesiaTTSService(api_key=os.getenv("CARTESIA_API_KEY"), voice_id=config.get("tts_voice_id") or "eda_id")
    return stt, llm, tts, system_messages

# --- PIPECAT RAW TCP İÇİN ÖZEL GİRDİ/ÇIKTI ---
class RawTCPOutput(FrameProcessor):
    def __init__(self, writer):
        super().__init__()
        self.writer = writer

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame):
            self.writer.write(frame.audio)
            await self.writer.drain()
        elif isinstance(frame, EndFrame):
            self.writer.close()

async def raw_tcp_input(reader):
    try:
        while True:
            data = await reader.read(320)
            if not data:
                yield EndFrame()
                break
            yield AudioRawFrame(audio=data, sample_rate=8000, num_channels=1)
    except Exception as e:
        logger.error(f"Input hata: {e}")
        yield EndFrame()

# --- CLAUDE'UN DOĞRULADIĞI HANDSHAKE İLE GÜNCELLENEN TCP LİSTENER ---
async def handle_fs_connection(reader, writer):
    addr = writer.get_extra_info('peername')
    logger.info(f"FreeSWITCH bağlandı: {addr}")
    try:
        # 1. FreeSWITCH'in ilk paketini oku (CHANNEL_DATA event)
        buf = b""
        while b"\n\n" not in buf:
            chunk = await reader.read(4096)
            if not chunk:
                raise Exception("Bağlantı handshake öncesi kapandı")
            buf += chunk
        logger.info("1. FreeSWITCH ilk paketi başarıyla okundu.")

        # 2. connect komutu gönder
        writer.write(b"connect\n\n")
        await writer.drain()
        logger.info("2. 'connect' komutu gönderildi.")

        # 3. connect cevabını oku
        buf = b""
        while b"\n\n" not in buf:
            chunk = await reader.read(4096)
            if not chunk:
                raise Exception("connect cevabı gelmedi")
            buf += chunk
        logger.info("3. 'connect' cevabı alındı.")

        # 4. myevents ile tüm eventleri al
        writer.write(b"myevents\n\n")
        await writer.drain()
        logger.info("4. 'myevents' emri gönderildi, Handshake TAMAMLANDI!")

        # --- BURADAN SONRA PIPECAT PIPELINE BAŞLATILIR ---
        config = {"llm_provider": "openai", "llm_model": "gpt-4o", "system_prompt": "Sen yardımsever bir asistansın.", "tts_provider": "cartesia"}
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
        await task.queue_frame(LLMMessagesFrame(system_messages))
        await runner.run(task)
        
    except Exception as e:
        logger.error(f"Bağlantı koptu veya hata ({addr}): {e}")
    finally:
        writer.close()
        await writer.wait_closed()

async def start_raw_tcp_server():
    server = await asyncio.start_server(handle_fs_connection, '0.0.0.0', 9001)
    logger.info("🚀 Raw TCP Server 9001 portunda aktif ve bekliyor...")
    async with server:
        await server.serve_forever()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(start_raw_tcp_server())

# --- API ENDPOINTS ---
@app.get("/")
async def root(): return {"status": "OK"}

@app.get("/health")
async def health(): return {"status": "healthy"}

@app.post("/outbound-call")
async def start_outbound_call(request: OutboundCallRequest, background_tasks: BackgroundTasks):
    call_id = str(uuid.uuid4())
    active_call_configs[call_id] = request.dict()

    originate_cmd = (
        f"{{"
        f"origination_caller_id_number={request.caller_id},"
        f"origination_caller_id_name={request.caller_id},"
        f"pipecat_call_id={call_id},"
        f"ignore_early_media=true,"
        f"progress_timeout=60,"
        f"absolute_codec_string=PCMA"
        f"}}sofia/gateway/netgsm/{request.phone_number} "
        f"&socket(10.0.1.7:9001 async full)"
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
