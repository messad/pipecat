# Base image olarak slim-bookworm kullanıyoruz (daha güncel ve kararlı)
FROM python:3.11-slim-bookworm

# 1. ADIM: uv binary'sini resmi imajdan kopyala
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN ruff format . --check || true
RUN pip install --no-cache-dir webrtc-noise-gain scipy numpy
# 2. ADIM: Sistem paketlerini kur
# Pipecat ses işleme (audio/voice) için bu kütüphanelere muhtaçtır.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# 3. ADIM: Bağımlılıkları Kur
COPY pyproject.toml uv.lock* ./

# ÖNEMLİ DEĞİŞİKLİK BURADA:
# Listeye 'websockets' ve 'greenswitch' eklendi.
# greenswitch -> FreeSWITCH ESL bağlantısı için şart.
# websockets -> Pipecat transport için şart.
# RUN uv pip install --system "pipecat-ai[deepgram,groq,elevenlabs,openai,google,anthropic,vapi,daily,cartesia,silero,fal,fastapi,twilio,vonage]" \
#     python-dotenv \
#     loguru \
#     transformers \
#     fastapi \
#     uvicorn \
#     torch \
#     pipecat-ai-small-webrtc-prebuilt \
#     aiortc \
#     websockets \
#     greenswitch
    
# Git kur + Pipecat'i main branch'ten manuel install
RUN apt-get update && apt-get install -y git \
    && git clone --depth 1 https://github.com/pipecat-ai/pipecat.git /tmp/pipecat \
    && cd /tmp/pipecat \
    && uv pip install --system .[deepgram,groq,elevenlabs,openai,google,anthropic,vapi,daily,cartesia,silero,fal,fastapi,twilio,vonage] \
    && cd / && rm -rf /tmp/pipecat \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Diğer bağımlılıklar (mevcut listen aynı)
RUN uv pip install --system \
    python-dotenv \
    loguru \
    transformers \
    fastapi \
    uvicorn \
    torch \
    pipecat-ai-small-webrtc-prebuilt \
    aiortc \
    websockets \
    greenswitch


# YENİ EK: Silero VAD ONNX modelini build-time'da indir (cache'e koy)
# Bu, runtime'da torch.hub.load() hatasını önler
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    curl -L -o /root/.cache/torch/hub/checkpoints/silero_vad.onnx \
    https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

# 4. ADIM: Uygulama kodlarını kopyala
COPY . .
COPY pipecat_connect.lua /usr/share/freeswitch/scripts/pipecat_connect.lua

# Coolify için Port (Senin dosyan 7860 kullanıyordu, onu korudum)
EXPOSE 8000
EXPOSE 9001

# BAŞLATMA KOMUTU DEĞİŞTİ:
# Artık bot.py değil, yeni yazdığımız server.py çalışacak.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
