import os
import sys
import uuid
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, BackgroundTasks
from pydantic import BaseModel
from loguru import logger
import uvicorn

# Pipecat importları
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import LLMUserResponseAggregator
from pipecat.services.openai import OpenAILLMService
from pipecat.services.anthropic import AnthropicLLMService 
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.transports.network.websocket_server import WebsocketServerTransport, WebsocketServerParams

# Loglama
logger.remove()
logger.add(sys.stderr, level="DEBUG")

app = FastAPI()

# Çağrı konfigürasyonlarını hafızada tutacağız
active_call_configs: Dict[str, Any] = {}

class OutboundCallRequest(BaseModel):
    phone_number: str
    llm_provider: str = "openai" 
    llm_model: str = "gpt-4o"
    stt_provider: str = "deepgram"
    tts_provider: str = "cartesia"
    system_prompt: str = "Sen yardımsever bir asistansın."

# --- DİNAMİK SERVİS FABRİKASI ---
def service_factory(config: dict):
    # 1. STT
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # 2. LLM
    if config.get("llm_provider") == "anthropic":
        llm = AnthropicLLMService(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=config.get("llm_model", "claude-3-5-sonnet-20240620")
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
            voice_id="21m00Tcm4TlvDq8ikWAM"
        )
    else:
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="a0e99841-438c-4a64-b679-ae501e7d6091"
        )
        
    return stt, llm, tts

# --- API ENDPOINT (n8n İÇİN) ---
@app.post("/outbound-call")
async def start_outbound_call(request: OutboundCallRequest, background_tasks: BackgroundTasks):
    call_id = str(uuid.uuid4())
    active_call_configs[call_id] = request.dict()
    
    logger.info(f"Arama isteği kuyruklandı. ID: {call_id}")
    
    # WebSocket URL'ini dönüyoruz ki FreeSWITCH'e bunu iletebilelim
    # Coolify Domain adresi buraya gelecek
    return {
        "status": "queued", 
        "call_id": call_id, 
        "websocket_path": f"/ws/{call_id}"
    }

# --- WEBSOCKET ENDPOINT (FREESWITCH İÇİN) ---
@app.websocket("/ws/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str):
    await websocket.accept()
    logger.info(f"FreeSWITCH bağlandı: {call_id}")

    config = active_call_configs.get(call_id, {})
    
    # FreeSWITCH genelde 8000Hz L16 formatında ses yollar
    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            audio_in_sample_rate=8000, 
            audio_out_sample_rate=8000,
            add_wav_header=False
        )
    )

    stt, llm, tts = service_factory(config)

    pipeline = Pipeline([
        transport.input(),
        stt,
        LLMUserResponseAggregator(),
        llm,
        tts,
        transport.output()
    ])

    task = PipelineTask(pipeline)
    await transport.start(websocket)
    await task.run()

    if call_id in active_call_configs:
        del active_call_configs[call_id]
