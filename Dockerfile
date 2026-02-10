FROM python:3.11-slim

# Sistem paketleri: audio/voice için zorunlu, build için gcc
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ffmpeg libsndfile1 libportaudio2 build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv'yi manuel kur (installer script + PATH export + versiyon pin ile stabil)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && echo 'export PATH="/root/.cargo/bin:$PATH"' >> /root/.bashrc \
    && /root/.cargo/bin/uv --version  # test et (log'larda göreceksin)

# PATH'i kalıcı set et (her layer'da çalışsın)
ENV PATH="/root/.cargo/bin:$PATH"

# Çalışma dizini
WORKDIR /app

# uv.lock ve pyproject.toml ile cache'le (eğer varsa)
COPY pyproject.toml uv.lock* ./

# venv oluştur + pipecat-ai'yi tüm extras ile kur (system-wide, no venv activate gerekmez)
RUN uv venv /opt/venv \
    && . /opt/venv/bin/activate \
    && uv pip install --system "pipecat-ai[deepgram,groq,elevenlabs,openai,google,anthropic,vapi,daily,cartesia,silero,fal,fastapi,twilio,vonage]" \
    && uv sync --frozen --no-dev --no-install-project

# Tüm kodu kopyala
COPY . .

# Env path venv için
ENV PATH="/opt/venv/bin:$PATH"

# Port (Pipecat runner default 7860)
EXPOSE 7860

# Bot'u çalıştır (bot dosyanı değiştir – örn. bot.py veya examples/voice-bot.py)
CMD ["python", "bot.py"]
