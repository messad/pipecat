# Base image olarak slim-bookworm kullanıyoruz (daha güncel ve kararlı)
FROM python:3.11-slim-bookworm

# 1. ADIM: uv binary'sini resmi imajdan kopyala (En güvenli yöntem budur)
# curl ile indirmek yerine bunu kullanmak path hatalarını %100 çözer.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

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
# uv.lock ve pyproject.toml dosyalarını kopyala
COPY pyproject.toml uv.lock* ./

# ÖNEMLİ: Container içinde venv oluşturmak yerine --system flag'i kullanıyoruz.
# Docker zaten izole bir ortam olduğu için venv aktivasyonu ile uğraşmanıza gerek yok.
# Bu komut paketleri doğrudan sistem python'una kurar.
RUN uv pip install --system "pipecat-ai[deepgram,groq,elevenlabs,openai,google,anthropic,vapi,daily,cartesia,silero,fal,fastapi,twilio,vonage]" \
    python-dotenv \
    loguru \
    transformers \
    fastapi \
    uvicorn \
    torch \
    pipecat-ai-small-webrtc-prebuilt

# Eğer elinizde bir uv.lock dosyası varsa ve sadece onu senkronize etmek isterseniz:
# RUN uv sync --frozen --system --no-dev

# 4. ADIM: Uygulama kodlarını kopyala
COPY . .

# Coolify için Port (Değiştirmediyseniz 7860)
EXPOSE 7860

# Ortam değişkenleri Coolify arayüzünden geleceği için burada ENV tanımlamaya gerek yok.
# Botu başlat
CMD ["python", "examples/quickstart/bot.py"]
