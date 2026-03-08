import os
import sys
import uuid
import asyncio
import logging
import base64
import json
from typing import Dict, Any, Optional
from deepgram import LiveOptions
from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from loguru import logger
import uvicorn

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s")
logging.getLogger("websockets.client").setLevel(logging.WARNING)
logging.getLogger("websockets.server").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.groq import GroqLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService, TextAggregationMode
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transcriptions.language import Language
from pipecat.frames.frames import (
    Frame,
    StartFrame,
    EndFrame,
    ErrorFrame,
    TranscriptionFrame,
    TextFrame,
    LLMMessagesUpdateFrame,
    InputAudioRawFrame,
    AudioRawFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    OutputAudioRawFrame,
	UserTurnEndedFrame,
)

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

logger.remove()
logger.add(sys.stderr, level="DEBUG")

app = FastAPI()

active_call_configs: Dict[str, Any] = {}

FS_HOST = os.getenv("FREESWITCH_HOST", "freeswitchcon")
FS_ESL_PORT = int(os.getenv("FREESWITCH_ESL_PORT", "8021"))
FS_ESL_PASSWORD = os.getenv("FREESWITCH_ESL_PASSWORD", "ClueCon")
PIPECAT_WS_BASE = os.getenv("PIPECAT_WS_BASE_URL", "ws://pipecatcon:8000")

# Türkçe dolgu kelimeleri — LLM cevap üretirken araya girer, gecikmeyi maskeler
TR_FILLER_PHRASES = [
    "Hmm...",
    "Tabii,",
    "Anladım,",
    "Hemen bakıyorum,",
    "Bir saniye,",
]


class OutboundCallRequest(BaseModel):
    phone_number: str
    caller_id: str = "2167064380"
    system_prompt: Optional[str] = "Sen yardımsever bir asistansın. Kısa ve net cevaplar ver."

    stt_provider: Optional[str] = "deepgram"
    stt_model: Optional[str] = "nova-3"
    stt_language: Optional[str] = "tr"
    stt_sample_rate: Optional[int] = 8000

    llm_provider: Optional[str] = "groq"
    llm_model: Optional[str] = "llama-3.3-70b-versatile"

    tts_provider: Optional[str] = "cartesia"
    tts_model: Optional[str] = "sonic-multilingual"   # ← postmandan gönderilebilir
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
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            live_options=LiveOptions(
                model=stt_model,
                language=stt_language,
                encoding="linear16",
                sample_rate=stt_sample_rate,
                channels=1,
                interim_results=True,   # Deepgram docs'ta önerilen: açık kalsın
                punctuate=True,
                smart_format=True,      # Türkçe akıllı formatlama
                profanity_filter=False,
            )
        )
    else:
        raise ValueError(f"Desteklenmeyen STT provider: {stt_provider}")

    system_prompt = config.get("system_prompt", "Sen yardımsever bir asistansın.")
    context = LLMContext(messages=[{"role": "system", "content": system_prompt}])

    vad_analyzer = SileroVADAnalyzer(
        sample_rate=stt_sample_rate,
        params=VADParams(
            stop_secs=0.75,             # Türkçe sondan eklemeli yapı için ideal
            start_secs=0.4,             # min_speech_duration karşılığı
            min_volume=0.6,
            confidence=0.8
        )
    )

    context_aggregator_pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad_analyzer
        ),
    )

    llm_provider = config.get("llm_provider", "groq")
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
    elif llm_provider == "groq":
        llm = GroqLLMService(
            api_key=os.getenv("GROQ_API_KEY"),
            model=config.get("llm_model", "llama-3.3-70b-versatile"),
			temperature=0.7,
			top_p=0.9
        )
    else:
        raise ValueError(f"Desteklenmeyen LLM provider: {llm_provider}")

    tts_provider = config.get("tts_provider", "cartesia")
    if tts_provider == "cartesia":
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id=config.get("tts_voice_id") or "39f753ef-b0eb-41cd-aa53-2f3c284f948f",
            sample_rate=8000,
            model=config.get("tts_model", "sonic-multilingual"),
            language=Language.TR,
            speed=0.95,
            emotion="friendly",
			text_aggregation_mode=TextAggregationMode.SENTENCE
            # text_aggregation_mode=TextAggregationMode.TOKEN     # düşük latency istersen
        )
    elif tts_provider == "elevenlabs":
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=config.get("tts_voice_id") or "21m00Tcm4TlvDq8ikWAM"
        )
    else:
        raise ValueError(f"Desteklenmeyen TTS provider: {tts_provider}")

    return stt, llm, tts, context_aggregator_pair


# ---------------------------------------------------------------------------
# DOLGU KELİMELERİ PROCESSOR
# LLM cevap üretmeye başlar başlamaz ilk token gelmeden önce kısa bir
# dolgu cümlesi TTS'e gönderilir. Gecikme hissini maskeler.
# ---------------------------------------------------------------------------
class FillerWordInjector(FrameProcessor):
    def __init__(self, phrases: list):
        super().__init__()
        self.phrases = phrases
        self._phrase_index = 0

    def _next_phrase(self) -> str:
        phrase = self.phrases[self._phrase_index % len(self.phrases)]
        self._phrase_index += 1
        return phrase

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserTurnEndedFrame):
            filler = self._next_phrase()
            logger.debug(f"💬 [Dolgu]: '{filler}'")
            await self.push_frame(TextFrame(text=filler), direction)
        await self.push_frame(frame, direction)


# --- SHARED STATE ---
class BotSpeakingState:
    def __init__(self):
        self.is_speaking = False


# --- UPSTREAM SUPPRESSOR (AEC) ---
class SoftwareEchoSuppressor(FrameProcessor):
    def __init__(self, state: BotSpeakingState):
        super().__init__()
        self.state = state
        self._speech_start_time = None
        self._min_interruption_secs = 0.5  # 800ms konuşmadan önce interrupt yok

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and self.state.is_speaking:
            now = asyncio.get_event_loop().time()
            if self._speech_start_time is None:
                self._speech_start_time = now
            elapsed = now - self._speech_start_time
            if elapsed < self._min_interruption_secs:
                return  # henüz yeterince konuşmadı, drop
            # 500ms geçti, gerçek interrupt — mikrofonu aç
        else:
            self._speech_start_time = None
        await self.push_frame(frame, direction)


# --- DOWNSTREAM TRACKER ---
class BotSpeakingTracker(FrameProcessor):
    def __init__(self, state: BotSpeakingState):
        super().__init__()
        self.state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSStartedFrame):
            logger.info("🔇 [AEC] Asistan konuşmaya başladı, mikrofon kapatıldı.")
            self.state.is_speaking = True
        elif isinstance(frame, TTSStoppedFrame):
            logger.info("🔊 [AEC] Asistan sustu, mikrofon açıldı.")
            self.state.is_speaking = False
        await self.push_frame(frame, direction)


# --- FREESWITCH INPUT ---
class FreeSWITCHInputProcessor(FrameProcessor):
    def __init__(self, websocket: WebSocket):
        super().__init__()
        self.websocket = websocket
        self._receive_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            if not self._receive_task:
                logger.info("🟢 FreeSWITCHInputProcessor: WebSocket okuma döngüsü başlıyor...")
                self._receive_task = asyncio.create_task(self._receive_audio_loop())
        await self.push_frame(frame, direction)

    async def _receive_audio_loop(self):
        try:
            while True:
                message = await self.websocket.receive()
                if 'bytes' in message and message['bytes']:
                    audio_frame = InputAudioRawFrame(
                        audio=message['bytes'],
                        sample_rate=8000,
                        num_channels=1
                    )
                    await self.push_frame(audio_frame, FrameDirection.DOWNSTREAM)
                elif message.get('type') == 'websocket.disconnect':
                    logger.info("FreeSWITCHInputProcessor: WebSocket kapandı.")
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"FreeSWITCH WebSocket okuma hatası: {e}")
        finally:
            await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    async def cleanup(self):
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            logger.info("🧹 FreeSWITCHInputProcessor: Temizlendi.")


# --- WEBSOCKET OUTPUT ---
class WebSocketOutput(FrameProcessor):
    def __init__(self, websocket: WebSocket):
        super().__init__()
        self.websocket = websocket
        self._log_counter = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, (AudioRawFrame, TTSAudioRawFrame, OutputAudioRawFrame)) and not isinstance(frame, InputAudioRawFrame):
            try:
                payload = json.dumps({
                    "type": "streamAudio",
                    "data": {
                        "audioDataType": "raw",
                        "sampleRate": 8000,
                        "audioData": base64.b64encode(frame.audio).decode("utf-8")
                    }
                })
                await self.websocket.send_text(payload)
                self._log_counter += 1
                if self._log_counter <= 5 or self._log_counter % 100 == 0:
                    logger.info(f"🔊 [SES GÖNDERİLDİ] {len(frame.audio)} byte (Paket #{self._log_counter})")
            except Exception as e:
                err = str(e).lower()
                if "closed" in err or "disconnect" in err or "runtime" in err:
                    logger.warning("FreeSWITCH bağlantısı kapandı.")
                else:
                    logger.error(f"WebSocket ses gönderme hatası: {e}")
        await self.push_frame(frame, direction)


# --- DEBUG LOGGERS ---
class STTLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            logger.info(f"🎙️ [STT]: {frame.text}")
        await self.push_frame(frame, direction)


class LLMLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            logger.debug(f"🧠 [LLM token]: {frame.text}")
        await self.push_frame(frame, direction)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket bağlantısı kuruldu.")

    call_id = websocket.query_params.get("call_id")
    if not call_id or call_id not in active_call_configs:
        logger.error(f"Geçersiz call_id: {call_id}")
        await websocket.close(code=1008, reason="Geçersiz veya eksik call_id")
        return

    config = active_call_configs[call_id]
    logger.info(f"Config bulundu. call_id={call_id}")

    stt, llm, tts, context_aggregator_pair = service_factory(config)

    bot_state = BotSpeakingState()
    echo_suppressor = SoftwareEchoSuppressor(bot_state)
    bot_tracker = BotSpeakingTracker(bot_state)
    fs_input = FreeSWITCHInputProcessor(websocket)
    fs_output = WebSocketOutput(websocket)
    stt_logger = STTLogger()
    llm_logger = LLMLogger()
    filler_injector =  FillerWordInjector(TR_FILLER_PHRASES)

    pipeline = Pipeline([
        fs_input,
        echo_suppressor,
        stt,
        stt_logger,
        context_aggregator_pair.user(),
        llm,
        llm_logger,
        filler_injector,    # LLM'in ilk tokeni gelince dolgu kelimesi TTS'e gider
        tts,                # Cartesia SENTENCE modunda çalışıyor, ayrıca buffer gerekmez
        bot_tracker,
        fs_output,
        context_aggregator_pair.assistant()
    ])

    runner = PipelineRunner()
    task = PipelineTask(pipeline)

    # Cold start kaldırıldı — system_prompt yeterli, ajan kullanıcıyı bekler

    async def run_pipeline():
        await runner.run(task)

    logger.info("Pipeline başlatılıyor...")
    try:
        await run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline hatası: {e}")
    finally:
        await fs_input.cleanup()
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
        f"pipecat_ws_url={ws_url},"
        f"ignore_early_media=true,"
        f"progress_timeout=60,"
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
