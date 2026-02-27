import json
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
from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect

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

# --- PİPECAT WEBSOCKET İÇİN ÖZEL GİRDİ/ÇIKTI ---
class WebSocketOutput(FrameProcessor):
    def __init__(self, websocket: WebSocket):
        super().__init__()
        self.websocket = websocket

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame):
            try:
                # TTS'ten gelen L16 8000Hz sesi WebSocket üzerinden FreeSWITCH'e gönder
                await self.websocket.send_bytes(frame.audio)
            except Exception as e:
                logger.error(f"WebSocket ses gönderme hatası: {e}")
        elif isinstance(frame, EndFrame):
            pass # Bağlantıyı FastAPI route yönetecek

async def websocket_input(websocket: WebSocket):
    try:
        while True:
            # FreeSWITCH'ten gelen ses paketlerini (bytes) oku
            data = await websocket.receive_bytes()
            yield AudioRawFrame(audio=data, sample_rate=8000, num_channels=1)
    except WebSocketDisconnect:
        logger.info("FreeSWITCH WebSocket bağlantısı koptu.")
        yield EndFrame()
    except Exception as e:
        logger.error(f"WebSocket okuma hatası: {e}")
        yield EndFrame()

# --- ADIM 2: YENİ WEBSOCKET LİSTENER ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("FreeSWITCH WebSocket üzerinden bağlandı.")
    
    try:
        # FreeSWITCH ilk bağlantıda metadata'yı metin (text) olarak gönderir
        metadata_str = await websocket.receive_text()
        meta = json.loads(metadata_str)
        logger.info(f"Metadata alındı, ses akışı başlıyor: {meta}")

        # Pipeline Başlatılıyor
        config = {
            "llm_provider": "openai", 
            "llm_model": "gpt-4o", 
            "system_prompt": "Sen yardımsever bir asistansın.", 
            "tts_provider": "cartesia"
        }
        stt, llm, tts, system_messages = service_factory(config)
        output_processor = WebSocketOutput(websocket)

        pipeline = Pipeline([
            websocket_input(websocket),
            stt,
            LLMUserResponseAggregator(),
            llm,
            tts,
            output_processor
        ])

        runner = PipelineRunner()
        task = PipelineTask(pipeline)
        
        await task.queue_frame(LLMMessagesFrame(system_messages))
        logger.info("Pipeline çalışıyor...")
        await runner.run(task)
        
    except WebSocketDisconnect:
        logger.info("WebSocket bağlantısı (aramadan dolayı) sonlandı.")
    except Exception as e:
        logger.error(f"WebSocket işleme hatası: {e}")
    finally:
        # Bağlantı kapanmışsa bile kapatmayı dener, hata verirse yoksayar
        try:
            await websocket.close()
        except:
            pass

async def start_raw_tcp_server():
    server = await asyncio.start_server(handle_fs_connection, '0.0.0.0', 9001)
    logger.info("🚀 Raw TCP Server 9001 portunda aktif ve bekliyor...")
    async with server:
        await server.serve_forever()

# --- API ENDPOINTS ---
@app.get("/")
async def root(): return {"status": "OK"}

@app.get("/health")
async def health(): return {"status": "healthy"}

@app.post("/outbound-call")
async def start_outbound_call(request: OutboundCallRequest, background_tasks: BackgroundTasks):
    call_id = str(uuid.uuid4())
    active_call_configs[call_id] = request.dict()

    # YENİ EKLENEN KISIM: api_on_answer ve ws:// yönlendirmesi
    originate_cmd = (
        f"{{"
        f"origination_uuid={call_id},"
        f"origination_caller_id_number={request.caller_id},"
        f"origination_caller_id_name=AI_Asistan,"
        f"pipecat_call_id={call_id},"
        f"ignore_early_media=true,"
        f"progress_timeout=60,"
        f"absolute_codec_string=PCMU,"
        f"api_on_answer='uuid_audio_stream {call_id} start ws://10.0.1.7:8000/ws mono 8000'" 
        f"}}sofia/gateway/netgsm/{request.phone_number} "
        f"&park()" 
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
