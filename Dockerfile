# Slim base, Python 3.11 (Pipecat uyumlu, düşük latency için hafif)
FROM python:3.11-slim

# Sistem deps kur (AI/voice için zorunlu: ffmpeg audio, libsndfile ses işleme, curl uv için)
RUN apt-get update && apt-get install -y \
    curl ffmpeg libsndfile1 libportaudio2 build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv kur (Pipecat resmi önerisi, hızlı install için)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Çalışma dizini
WORKDIR /app

# Bağımlılık dosyalarını kopyala (cache için)
COPY pyproject.toml uv.lock* ./

# Virtual env + tüm extras ile kur (en düşük latency/performans için system install, no-dev)
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install --system "pipecat-ai[deepgram,groq,elevenlabs,openai,google,anthropic,vapi,daily,cartesia,silero,fal,fastapi,twilio,vonage]" \
    && uv sync --frozen --no-dev

# Tüm kodu kopyala
COPY . .

# Env path
ENV PATH="/app/.venv/bin:$PATH"

# Port expose (7860 Pipecat default)
EXPOSE 7860

# Bot'u çalıştır (bot dosyanı değiştir, örn. examples/quickstart/bot.py)
CMD ["python", "examples/quickstart/bot.py"]
