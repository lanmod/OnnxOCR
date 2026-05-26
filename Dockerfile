FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV ONNXOCR_PORT=5005

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

COPY . .

EXPOSE 5005

CMD ["python", "app-service.py"]
