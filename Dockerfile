FROM python:3.11-slim

# ffmpeg 用于音轨提取与容器转换；faster-whisper 走 PyAV 解码，两者都备上更稳
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py pipeline.py prompts.py ./
COPY static ./static

# 语音模型缓存放卷里，重启不重新下载
ENV HF_HOME=/data/models \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_DISABLE_XET=1 \
    PORT=8790

VOLUME ["/data"]

EXPOSE 8790

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8790/api/health',timeout=4)"

CMD ["python", "app.py"]
