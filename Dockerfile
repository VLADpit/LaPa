FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8501 \
    HOME=/home/appuser \
    PYTHONPATH=/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.0+cpu \
        torchvision==0.17.0+cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /root/.EasyOCR/model && \
    python -c "import easyocr; easyocr.Reader(['en', 'ru'], gpu=False, verbose=False)"

RUN groupadd --system --gid 1000 appuser && \
    useradd --system --uid 1000 --gid appuser --home /home/appuser appuser && \
    mkdir -p /home/appuser && \
    if [ -d /root/.EasyOCR ]; then mv /root/.EasyOCR /home/appuser/.EasyOCR; fi && \
    chown -R appuser:appuser /home/appuser /app


COPY . .

RUN chown -R appuser:appuser /app && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app/data

USER appuser

EXPOSE ${PORT}

CMD streamlit run app/main.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false