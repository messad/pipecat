import os
import sys
import uuid
import asyncio
import logging
import base64
import json
import re
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
from pipecat.services.cartesia.tts import CartesiaTTSService
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
                interim_results=False,
                punctuate=True,
                smart_format=True,
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
            stop_secs=0.5,
            start_secs=0.3,
            min_volume=0.6,
            confidence=0.75
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
            model=config.get("llm_model", "llama-3.3-70b-versatile")
        )
    else:
        raise ValueError(f"Desteklenmeyen LLM provider: {llm_provider}")

    tts_provider = config.get("tts_provider", "cartesia")
    if tts_provider == "cartesia":
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id=config.get("tts_voice_id") or "39f753ef-b0eb-41cd-aa53-2f3c284f948f",
            sample_rate=8000,
            model="sonic-multilingual",
            language=Language.TR,
            speed=1.0,
            emotion="friendly",
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
# TÜRKÇE OPTİMİZE SENTENCE BUFFER
# ---------------------------------------------------------------------------
class SentenceBuffer(FrameProcessor):
    TR_ABBREVIATIONS = {
        "dr", "prof", "doç", "öğr", "müh", "yrd", "uzm", "arş",
        "sok", "cad", "apt", "blv", "no", "tel", "faks", "www",
        "kg", "km", "cm", "mm", "lt", "ml", "vs", "vb", "bkz",
        "mrk", "ünv", "a.ş", "ltd", "şti",
    }

    HARD_END = re.compile(r'[.!?…]+')

    TR_CONNECTOR_PAUSE = re.compile(
        r',\s*(ancak|fakat|lakin|ama|oysa|halbuki|bunun\s+için|bu\s+nedenle|'
        r'bu\s+yüzden|dolayısıyla|sonuç\s+olarak|öte\s+yandan|'
        r'bir\s+taraftan|diğer\s+taraftan|ayrıca|dahası|üstelik|'
        r'buna\s+ek\s+olarak|bunun\s+yanında|örneğin|mesela|'
        r'yani|kısacası|özetle|sonuçta)\s',
        re.IGNORECASE
    )

    def __init__(self, min_chars: int = 8):
        super().__init__()
        self.buffer = ""
        self.min_chars = min_chars

    def _is_abbreviation(self, text: str, dot_pos: int) -> bool:
        before = text[:dot_pos].rstrip()
        word = before.split()[-1].lower() if before.split() else ""
        if word.isdigit():
            return True
        if len(word) == 1:
            return True
        if word.rstrip('.') in self.TR_ABBREVIATIONS:
            return True
        return False

    def _find_sentence_boundary(self, text: str) -> int:
        for match in self.HARD_END.finditer(text):
            pos = match.end()
            if match.group().startswith('.') and self._is_abbreviation(text, match.start()):
                continue
            if pos >= len(text) or text[pos] in (' ', '\n', '\t'):
                return pos
            if pos == len(text):
                return pos

        connector_match = self.TR_CONNECTOR_PAUSE.search(text)
        if connector_match and connector_match.start() > self.min_chars:
            return connector_match.start() + 1

        if len(text) >= 40:
            last_comma = text.rfind(',')
            if last_comma > self.min_chars:
                return last_comma + 1

        return -1

    async def _flush(self, text: str):
        text = text.strip()
        if len(text) >= self.min_chars:
            logger.debug(f"📝 [SentenceBuffer → TTS]: '{text}'")
            await self.push_frame(TextFrame(text=text))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            self.buffer += frame.text
            while True:
                boundary = self._find_sentence_boundary(self.buffer)
                if boundary == -1:
                    break
                sentence = self.buffer[:boundary]
                self.buffer = self.buffer[boundary:].lstrip()
                await self._flush(sentence)

        elif isinstance(frame, (EndFrame, LLMMessagesUpdateFrame)):
            if self.buffer.strip():
                await self._flush(self.buffer)
                self.buffer = ""
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)


# --- SHARED STATE ---
class BotSpeakingState:
    def __init__(self):
        self.is_speaking = False


# --- UPSTREAM SUPPRESSOR ---
class SoftwareEchoSuppressor(FrameProcessor):
    def __init__(self, state: BotSpeakingState):
        super().__init__()
        self.state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and self.state.is_speaking:
            return
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
                if "closed" in str(e).lower() or "disconnect" in str(e).lower():
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
    sentence_buffer = SentenceBuffer(min_chars=8)

    pipeline = Pipeline([
        fs_input,
        echo_suppressor,
        stt,
        stt_logger,
        context_aggregator_pair.user(),
        llm,
        llm_logger,
        sentence_buffer,        # LLM tokenlarını biriktir, cümle tamamlanınca TTS'e gönder
        tts,
        bot_tracker,
        fs_output,
        context_aggregator_pair.assistant()
    ])

    runner = PipelineRunner()
    task = PipelineTask(pipeline)

    initial_message = LLMMessagesUpdateFrame(
        messages=[{
            "role": "user",
            "content": "Telefon bağlandı. Kısa, enerjik ve kibar bir şekilde Türkçe 'Merhaba, size nasıl yardımcı olabilirim?' de."
        }],
        run_llm=True
    )

    async def run_pipeline():
        await runner.run(task)

    async def send_initial():
        await asyncio.sleep(0.5)
        logger.info("Cold Start mesajı gönderiliyor...")
        await task.queue_frame(initial_message)

    logger.info("Pipeline başlatılıyor...")
    try:
        await asyncio.gather(run_pipeline(), send_initial())
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
