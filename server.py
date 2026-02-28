import json
import os
import sys
import uuid
import asyncio
import logging
from typing import Dict, Any, Optional

from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from loguru import logger
import uvicorn

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s")

# Pipecat Importlar (universal LLM context için güncel yol)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService  # sub-module (deprecation fix)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,  # YENİ EK: Bu hatayı çözer
)
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.frames.frames import LLMMessagesFrame, AudioRawFrame, EndFrame, ErrorFrame

# YENİ VAD IMPORTLARI
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from deepgram import LiveOptions  # Deepgram live_options için

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

# ESL bağlantı (değişmedi)
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

# Servis factory (VAD EKLEME BURADA)
def service_factory(config: dict):
    # Deepgram STT - VAD'ı kapat, Silero'ya bırak
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        language="tr",
        live_options=LiveOptions(
            interim_results=True,
            vad_events=False,
            utterance_end_ms=800,
            endpointing=300,
            sample_rate=8000,
            channels=1,
            encoding="linear16"
        )
    )
    
    system_prompt = config.get("system_prompt", "Sen yardımsever bir asistansın.")
    context = LLMContext(
        messages=[{"role": "system", "content": system_prompt}]
    )
    
    # VAD ANALYZER EKLE
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            stop_secs=0.3,   # Sessizlik sonrası utterance bitir (düşük = daha hızlı cevap)
            start_secs=0.15  # Konuşma başlangıcı hassasiyeti
        )
    )
    
    context_aggregator_pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad_analyzer,
        ),
    )
    
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model=config.get("llm_model", "gpt-4o"))
    tts = CartesiaTTSService(api_key=os.getenv("CARTESIA_API_KEY"), voice_id=config.get("tts_voice_id") or "eda_id")
    
    return stt, llm, tts, context_aggregator_pair

# WebSocket Output (değişmedi)
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

# WebSocket Input generator (değişmedi)
async def websocket_input(websocket: WebSocket):
    try:
        while True:
            message = await websocket.receive()
            if 'text' in message and message['text']:
                logger.info(f"FreeSWITCH Metadata: {message['text']}")
            elif 'bytes' in message and message['bytes']:
                logger.info(f"Binary ses paketi geldi! Boyut: {len(message['bytes'])} bytes")
                yield AudioRawFrame(audio=message['bytes'], sample_rate=8000, num_channels=1)
            elif message.get('type') == 'websocket.disconnect':
                logger.info("WebSocket disconnect yakalandı.")
                break
    except Exception as e:
        logger.error(f"WebSocket okuma hatası: {e}")
    yield EndFrame()

# WebSocket Endpoint (push_audio debug'li hali kaldı)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("FreeSWITCH WebSocket üzerinden bağlandı. Ses akışı başlıyor...")
    
    try:
        config = {
            "llm_provider": "openai", 
            "llm_model": "gpt-4o", 
            "system_prompt": "Sen yardımsever bir asistansın.", 
            "tts_provider": "cartesia"
        }
        
        stt, llm, tts, context_aggregator_pair = service_factory(config)
        output_processor = WebSocketOutput(websocket)

        pipeline = Pipeline([
            stt,
            context_aggregator_pair.user(),  # User input aggregator
            llm,
            tts,
            output_processor,
            context_aggregator_pair.assistant()  # Assistant output aggregator
        ])

        runner = PipelineRunner()
        task = PipelineTask(pipeline)
        await task.queue_frame(LLMMessagesFrame([{"role": "system", "content": config["system_prompt"]}]))
        
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
                logger.info("WebSocket kapandı (istemci tarafı).")
                await task.queue_frame(EndFrame())
            except Exception as e:
                logger.error(f"Ses push hatası: {e}")
                await task.queue_frame(ErrorFrame(f"Audio input error: {e}"))
            finally:
                await task.queue_frame(EndFrame())  # Garanti kapanış

        logger.info("Pipeline ve Ses Girişi başlatılıyor...")
        
        await asyncio.gather(runner.run(task), push_audio())
        
    except Exception as e:
        logger.error(f"WebSocket işleme hatası: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass

# API Endpoints (değişmedi)
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
