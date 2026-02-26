import os
import sys
import uuid
import asyncio
import socket
from typing import Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from loguru import logger
import uvicorn

# Pipecat pipeline bileşenleri
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai import OpenAILLMService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.cartesia import CartesiaTTSService
# ÖNEMLİ: Raw TCP üzerinden ses işleme için gerekli transport
from pipecat.transports.network.generic_tcp import GenericTCPTransport, GenericTCPTransportParams

logger.remove()
logger.add(sys.stderr, level="DEBUG")

app = FastAPI()

# Global çağrı konfigürasyonları
active_call_configs: Dict[str, Any] = {}

FS_HOST = os.getenv("FREESWITCH_HOST", "freeswitchcon")
FS_ESL_PORT = int(os.getenv("FREESWITCH_ESL_PORT", "8021"))
FS_ESL_PASSWORD = os.getenv("FREESWITCH_ESL_PASSWORD", "ClueCon")
# FreeSWITCH'in Pipecat'e ulaşacağı adres
PIPECAT_HOST = "0.0.0.0"
PIPECAT_TCP_PORT = 9001

class OutboundCallRequest(BaseModel):
    phone_number: str
    system_prompt: str = "Sen hızlı bir asistansın."
    caller_id: str = "2167064380"

async def esl_originate(originate_cmd: str):
    """FreeSWITCH'e outbound çağrı emri gönderir."""
    reader, writer = await asyncio.open_connection(FS_HOST, FS_ESL_PORT)
    try:
        # Auth süreci
        await reader.readuntil(b"auth/request\n\n")
        writer.write(f"auth {FS_ESL_PASSWORD}\n\n".encode())
        await writer.drain()
        
        resp = await reader.readuntil(b"\n\n")
        if b"+OK" not in resp:
            raise Exception(f"ESL Auth Hatası: {resp}")

        # Çağrıyı başlat
        cmd = f"bgapi originate {originate_cmd}\n\n"
        writer.write(cmd.encode())
        await writer.drain()
        return await reader.readuntil(b"\n\n")
    finally:
        writer.close()
        await writer.wait_closed()

async def start_pipecat_handler(reader, writer):
    """FreeSWITCH bağlandığında Pipecat pipeline'ını yönetir."""
    addr = writer.get_extra_info('peername')
    logger.info(f"FreeSWITCH'ten raw bağlantı geldi: {addr}")

    try:
        # 1. FreeSWITCH Outbound Socket Handshake
        # FreeSWITCH bağlandığında 'connect' mesajı gönderir
        data = await reader.readuntil(b"\n\n")
        
        # Call ID'yi headerlardan yakala (originate komutunda eklemiştik)
        # Pratik olması için son aktif çağrıyı alıyoruz veya header parse edilebilir
        call_id = "default" # Gerçek projede data içindeki pipecat_call_id parse edilir

        # 2. Pipecat Transport Kurulumu (Raw TCP)
        # FreeSWITCH genellikle 8000Hz Mono PCMA/L16 bekler
        transport = TCPTransport(
            reader, 
            writer,
            params=TCPTransportParams(
                audio_in_sample_rate=8000,
                audio_out_sample_rate=8000,
            )
        )

        # 3. Servislerin Hazırlanması
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"), language="tr")
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
        tts = CartesiaTTSService(api_key=os.getenv("CARTESIA_API_KEY"), voice_id="eda_id")

        messages = [{"role": "system", "content": "Sen hızlı bir sesli asistansın."}]
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])

        task = PipelineTask(pipeline)
        runner = PipelineRunner()
        
        logger.info("Pipeline başlatılıyor...")
        await runner.run(task)

    except Exception as e:
        logger.error(f"Handler Hatası: {e}")
    finally:
        writer.close()
        await writer.wait_closed()

@app.on_event("startup")
async def start_tcp_listener():
    """TCP Port 9001'i dinlemeye başlar."""
    server = await asyncio.start_server(start_pipecat_handler, PIPECAT_HOST, PIPECAT_TCP_PORT)
    logger.info(f"🚀 Raw TCP Server dinliyor: {PIPECAT_HOST}:{PIPECAT_TCP_PORT}")
    asyncio.create_task(server.serve_forever())

@app.post("/outbound-call")
async def outbound_call(request: OutboundCallRequest):
    # FreeSWITCH'e giden komutta /ws/ gibi URL yapılarını sildik.
    # Sadece IP:PORT veriyoruz.
    originate_cmd = (
        f"{{origination_caller_id_number={request.caller_id},"
        f"absolute_codec_string=PCMA,"
        f"ignore_early_media=true}}"
        f"sofia/gateway/netgsm/{request.phone_number} "
        f"&socket({os.getenv('PIPECAT_HOST_INTERNAL', 'pipecatcon')}:{PIPECAT_TCP_PORT} async full)"
    )
    
    try:
        result = await esl_originate(originate_cmd)
        return {"status": "success", "detail": result.decode()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
