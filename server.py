import os
import sys
import uuid
import asyncio
import logging
import base64
import json
import audioop
from typing import Dict, Any, Optional
from deepgram import AsyncDeepgramClient
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
from pipecat.services.openai.llm import OpenAILLMService, OpenAILLMSettings
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.groq import GroqLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService, DeepgramSTTSettings
from pipecat.services.cartesia.tts import CartesiaTTSService, TextAggregationMode, CartesiaTTSSettings, GenerationConfig
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    LLMUserAggregator,
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
	UserStoppedSpeakingFrame,
	VADUserStartedSpeakingFrame,
	InterruptionFrame,
	CancelFrame
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
HANGUP_KEYWORDS = [
    "güle güle", "görüşürüz", "hoşça kal", "hoşçakal",
    "kapat", "kapatıyorum", "kapatalım", "tamam kapat",
    "bay bay", "iyi günler", "iyi akşamlar", "iyi geceler",
    "teşekkürler görüşürüz", "çok teşekkürler hoşça kal"
]

# Türkçe dolgu kelimeleri — LLM cevap üretirken araya girer, gecikmeyi maskeler
TR_FILLER_PHRASES = [
    "Hmm...",
    "Tabii,",
    "Anladım,",
    "Hemen bakıyorum,",
    "Bir saniye,",
]

CANCEL_KEYWORDS = [
    "dur", "bekle", "devam et", "devam edelim", "iptal",
    "aslında", "hayır kapat", "kapatma"
]

class CallEndDetector(FrameProcessor):
    def __init__(self, call_id: str, esl_host: str, esl_port: int, esl_password: str):
        super().__init__()
        self.call_id = call_id
        self.esl_host = esl_host
        self.esl_port = esl_port
        self.esl_password = esl_password
        self._pending_hangup = False
        self._hangup_task = None
        self._task = None  # pipeline task referansı, endpoint'ten set edilecek

    async def _hangup_via_esl(self):
        try:
            reader, writer = await asyncio.open_connection(self.esl_host, self.esl_port)
            await reader.readuntil(b"auth/request\n\n")
            writer.write(f"auth {self.esl_password}\n\n".encode())
            await writer.drain()
            await reader.readuntil(b"\n\n")
            writer.write(f"api uuid_kill {self.call_id}\n\n".encode())
            await writer.drain()
            await reader.readuntil(b"\n\n")
            writer.close()
            await writer.wait_closed()
            logger.info("📵 [CallEnd] ESL hangup gönderildi.")
        except Exception as e:
            logger.error(f"ESL hangup hatası: {e}")

    async def _delayed_hangup(self):
        try:
            await asyncio.sleep(1.5)
            if self._pending_hangup:
                logger.info("📵 [CallEnd] Timer doldu, kapatılıyor.")
                await self._hangup_via_esl()
                if self._task:
                    await self._task.queue_frame(EndFrame())
        except asyncio.CancelledError:
            logger.info("📵 [CallEnd] Hangup iptal edildi.")
        finally:
            self._pending_hangup = False
            self._hangup_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.lower().strip()

            # pending_hangup varken — sadece iptal kontrolü yap, başka hiçbir şey tetikleme
            if self._pending_hangup:
                if any(kw in text for kw in CANCEL_KEYWORDS):
                    logger.info("📵 [CallEnd] Kullanıcı iptal etti.")
                    self._pending_hangup = False
                    if self._hangup_task:
                        self._hangup_task.cancel()
                        self._hangup_task = None
                    await self.push_frame(frame, direction)
                else:
                    # Veda veya başka kelime — timer zaten çalışıyor, frame'i yut
                    logger.debug(f"📵 [CallEnd] Hangup beklenirken yeni transcript yutuldu: '{text}'")
                return

            # Normal akış — kapatma kelimesi kontrolü
            if any(kw in text for kw in HANGUP_KEYWORDS):
                logger.info(f"📵 [CallEnd] Kapatma algılandı: '{frame.text}'")
                self._pending_hangup = True
                veda = LLMMessagesUpdateFrame(
                    messages=[{
                        "role": "user",
                        "content": (
                            "Kullanıcı görüşmeyi bitirmek istiyor. "
                            "Tek cümleyle nazikçe veda et. Kısa tut."
                        )
                    }],
                    run_llm=True
                )
                await self.push_frame(veda, direction)
                return

        # TTS bitti + hangup bekliyorsa timer başlat
        if isinstance(frame, TTSStoppedFrame) and self._pending_hangup:
            if not self._hangup_task or self._hangup_task.done():
                logger.info("📵 [CallEnd] TTS bitti, hangup timer başlıyor (1.5s).")
                self._hangup_task = asyncio.create_task(self._delayed_hangup())

        await self.push_frame(frame, direction)

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
            # live_options=LiveOptions(
            #     model=stt_model,
            #     language=stt_language,
            #     encoding="linear16",
            #     sample_rate=stt_sample_rate,
            #     channels=1,
            #     interim_results=True,   # Deepgram docs'ta önerilen: açık kalsın
            #     punctuate=True,
            #     smart_format=True,      # Türkçe akıllı formatlama
            #     profanity_filter=False,
            # )
            encoding="linear16",
            channels=1,
            sample_rate=stt_sample_rate,
            settings=DeepgramSTTSettings(
                model=stt_model,
                language=stt_language,
                interim_results=True,
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
            stop_secs=1.1,             # Türkçe sondan eklemeli yapı için ideal
            start_secs=0.30,             # min_speech_duration karşılığı
            min_volume=0.55,
            confidence=0.85
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
            settings=OpenAILLMSettings(
                model=config.get("llm_model", "gpt-4o-mini"), # gpt-4o en doğal Türkçe
                temperature=0.7,                      # doğal varyasyon için ideal
                top_p=0.9,
                max_tokens=512,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                # Türkçe için ekstra optimizasyon
                system_instruction="sayıları her zaman yazı ile türet yirmi beş gibi. Cümle başlangıcında ingilizce ile karıştırılabilecek kelimeleri kullanma örneğin Size nasıl yardımcı olabilirim? burda size ingilizce kelime ile karıştırılabilir. Cümlenin ilk kelimesi sadece Türkçede olan kelimeler olsun.",
                # Diğer ayarlar (isteğe bağlı)
                seed=42,                              # tekrarlanabilirlik için
                extra={"response_format": {"type": "text"}}  # sadece metin
            )
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
            settings=CartesiaTTSSettings(
                generation_config=GenerationConfig(
                    speed=0.92,
                    emotion="cheerful"
                ),    
                pronunciation_dict_id="pdict_JL3JcmhtjtKd7rkV2Fwt6a"
            ),
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
        if isinstance(frame, TranscriptionFrame):
            filler = self._next_phrase()
            logger.debug(f"💬 [Dolgu]: '{filler}'")
            # Dolguyu önce gönder
            await self.push_frame(TextFrame(text=filler), direction)
        # TranscriptionFrame'i HER ZAMAN downstream'e geçir
        await self.push_frame(frame, direction)


# --- SHARED STATE ---
class BotSpeakingState:
    def __init__(self):
        self.is_speaking = False

# class InstantBargeInHandler(FrameProcessor):
#     def __init__(self, state: BotSpeakingState):
#         super().__init__()
#         self.state = state
#         self._consecutive_frames = 0
#         self._threshold = 35
#         self._echo_window_secs = 2.0

#     async def process_frame(self, frame: Frame, direction: FrameDirection):
#         await super().process_frame(frame, direction)

#         if isinstance(frame, InputAudioRawFrame):
#             if self.state.is_speaking:
#                 elapsed = asyncio.get_event_loop().time() - self.state.started_at
#                 if elapsed < self._echo_window_secs:
#                     # Echo window — tamamen drop et, say bile
#                     self._consecutive_frames = 0
#                     return

#                 self._consecutive_frames += 1
#                 if self._consecutive_frames >= self._threshold:
#                     logger.info("⚡ [BARGE-IN] Gerçek interrupt, TTS kesiliyor.")
#                     self.state.is_speaking = False
#                     self._consecutive_frames = 0
#                     await self.push_frame(InterruptionFrame(), direction)
#                 else:
#                     await self.push_frame(frame, direction)
#                     return
#             else:
#                 self._consecutive_frames = 0

#         await self.push_frame(frame, direction)

class InstantBargeInHandler(FrameProcessor):
    def __init__(self, state: BotSpeakingState):
        super().__init__()
        self.state = state
        self._consecutive_frames = 0
        self._threshold = 12        # Gerçek konuşma için ardışık frame sayısı (8-20 arası test et)
        self._rms_min = 400         # Yankı/gürültü eşiği — log'lardan kalibre edeceğiz

    def _rms(self, audio: bytes) -> float:
        try:
            return audioop.rms(audio, 2)  # 16-bit PCM
        except:
            return 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            if self.state.is_speaking:
                rms = self._rms(frame.audio)
                if rms < self._rms_min:
                    # Ses çok sessiz — yankı veya gürültü, drop et
                    self._consecutive_frames = 0
                    return

                self._consecutive_frames += 1
                if self._consecutive_frames >= self._threshold:
                    logger.info(f"⚡ [BARGE-IN] Gerçek ses algılandı (RMS:{rms:.0f}), TTS kesiliyor.")
                    self.state.is_speaking = False
                    self._consecutive_frames = 0
                    await self.push_frame(InterruptionFrame(), direction)
                else:
                    await self.push_frame(frame, direction)
                    return
            else:
                self._consecutive_frames = 0

        await self.push_frame(frame, direction)

# --- DOWNSTREAM TRACKER ---
# class BotSpeakingTracker(FrameProcessor):
#     def __init__(self, state: BotSpeakingState):
#         super().__init__()
#         self.state = state

#     async def process_frame(self, frame: Frame, direction: FrameDirection):
#         await super().process_frame(frame, direction)
#         if isinstance(frame, TTSStartedFrame):
#             self.state.is_speaking = True
#             self.state.started_at = asyncio.get_event_loop().time()
#             logger.info("🔇 [AEC] Mikrofon kapatıldı.")
#         elif isinstance(frame, TTSStoppedFrame):
#             self.state.is_speaking = False
#             logger.info("🔊 [AEC] Mikrofon açıldı.")
#         await self.push_frame(frame, direction)

# --- DOWNSTREAM TRACKER ---
class BotSpeakingTracker(FrameProcessor):
    def __init__(self, state: BotSpeakingState):
        super().__init__()
        self.state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSStartedFrame):
            self.state.is_speaking = True
            logger.info("🔇 [AEC] Mikrofon kapatıldı.")
        elif isinstance(frame, TTSStoppedFrame):
            self.state.is_speaking = False
            logger.info("🔊 [AEC] Mikrofon açıldı.")
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
    barge_in_handler = InstantBargeInHandler(bot_state)  # SoftwareEchoSuppressor + FastBargeInHandler birleşti
    bot_tracker = BotSpeakingTracker(bot_state)
    fs_input = FreeSWITCHInputProcessor(websocket)
    fs_output = WebSocketOutput(websocket)
    stt_logger = STTLogger()
    llm_logger = LLMLogger()
    filler_injector = FillerWordInjector(TR_FILLER_PHRASES)

    call_end_detector = CallEndDetector(
        call_id=call_id,
        esl_host=FS_HOST,
        esl_port=FS_ESL_PORT,
        esl_password=FS_ESL_PASSWORD
    )

    pipeline = Pipeline([
        fs_input,
        barge_in_handler,           # echo_suppressor + barge-in birleşik
        stt,
        stt_logger,
        call_end_detector,          # kapatma kelimesi kontrolü
        context_aggregator_pair.user(),
        llm,
        llm_logger,
        filler_injector,
        tts,
        bot_tracker,
        fs_output,
        context_aggregator_pair.assistant()
    ])

    runner = PipelineRunner()
    task = PipelineTask(pipeline)
    call_end_detector._task = task  # task referansını ver, delayed_hangup içinde kullanılacak

    async def run_pipeline():
        await runner.run(task)

    async def warmup_tts():
        await asyncio.sleep(1.0)
        await task.queue_frame(TextFrame(text=" "))
        logger.info("🔥 TTS warm-up frame gönderildi.")

    logger.info("Pipeline başlatılıyor...")
    try:
        await asyncio.gather(run_pipeline(), warmup_tts())
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
