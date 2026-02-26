import os
import sys
import uuid
import asyncio
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from loguru import logger
import uvicorn

# Pipecat 0.0.103 Uyumlu Importlar
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.aggregators.llm_response import LLMUserResponseAggregator
from pipecat.services.openai import OpenAILLMService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.frames.frames import LLMMessagesFrame, AudioRawFrame, EndFrame

logger.remove()
logger.add(sys.stderr, level="DEBUG")

app = FastAPI()
active_call_configs: Dict[str, Any] = {}

# --- CUSTOM RAW TCP TRANSPORT (0.0.103 İÇİN ÖZEL) ---
class RawTCPTransport:
    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer
        self.running = True

    async def input(self):
        """FreeSWITCH'ten gelen L16 sesi oku ve Pipeline'a gönder"""
        try:
            while self.running:
                # 20ms'lik L16 ses verisi (8000Hz * 2byte * 0.02s = 320 byte)
                data = await self.reader.read(320)
                if not data:
                    yield EndFrame()
                    break
                yield AudioRawFrame(data, sample_rate=8000, channels=1)
        except Exception as e:
            logger.error(f"Input hatası: {e}")
            yield EndFrame()

    async def output(self, frame):
        """Pipeline'dan gelen TTS sesini FreeSWITCH'e yaz"""
        if isinstance(frame, AudioRawFrame):
            self.writer.write(frame.data)
            await self.writer.drain()
        elif isinstance(frame, EndFrame):
            self.running = False
            self.writer.close()

# --- SERVİS FABRİKASI ---
def service_factory(config: dict):
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"), language="tr")
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model=config.get("llm_model", "gpt-4o"))
    tts = CartesiaTTSService(api_key=os.getenv("CARTESIA_API_KEY"), voice_id=config.get("tts_voice_id") or "eda_id")
    system_messages = [{"role": "system", "content": config.get("system_prompt", "Yardımsever bir asistansın.")}]
    return stt, llm, tts, system_messages

# --- RAW TCP SERVER (9001 PORTU) ---
async def start_raw_tcp_server():
    # server = await asyncio.start_server(handle_fs_connection, '0.0.0.0', 9001)
    server = await asyncio.start_server(handle_fs_connection, host=None, port=9001)
    logger.info("🚀 Raw TCP Server 9001 portunda dinliyor...")
    async with server:
        await server.serve_forever()

async def handle_fs_connection(reader, writer):
    addr = writer.get_extra_info('peername')
    logger.info(f"FreeSWITCH bağlandı: {addr}")

    # FreeSWITCH outbound el sıkışması (connect mesajını atla)
    await reader.readuntil(b"\n\n")

    transport = RawTCPTransport(reader, writer)
    # Varsayılan konfigürasyon (Daha sonra call_id ile geliştirilebilir)
    stt, llm, tts, system_messages = service_factory({})

    pipeline = Pipeline([
        transport.input(),  # Kaynak
        stt,
        LLMUserResponseAggregator(),
        llm,
        tts,
        transport.output   # Hedef (Output bir fonksiyon olarak verilir)
    ])

    task = PipelineTask(pipeline)
    runner = PipelineRunner()
    
    # İlk mesajı gönder
    await task.queue_frame(LLMMessagesFrame(system_messages))
    
    logger.info("Pipeline başlatıldı.")
    await runner.run(task)
    logger.info("Pipeline sonlandı.")

# --- API ENDPOINTS ---
class OutboundCallRequest(BaseModel):
    phone_number: str
    caller_id: str = "2167064380"

@app.on_event("startup")
async def startup_event():
    # TCP Server'ı arka planda başlat
    asyncio.create_task(start_raw_tcp_server())

@app.post("/outbound-call")
async def outbound_call(request: OutboundCallRequest):
    # FreeSWITCH'e verilen komut (Öncekiyle aynı, sadece IP/Port önemli)
    # Pipecatcon:9001 senin docker network ismin olmalı
    originate_cmd = (
        f"{{origination_caller_id_number={request.caller_id},"
        f"absolute_codec_string=PCMA,ignore_early_media=true}}"
        f"sofia/gateway/netgsm/{request.phone_number} "
        f"&socket(pipecatcon:9001 async full)"
    )
    # Burada mevcut esl_originate fonksiyonunu çağırabilirsin
    logger.info(f"Arama emri gönderiliyor: {originate_cmd}")
    return {"status": "initiated", "cmd": originate_cmd}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
