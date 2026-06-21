FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl libpq-dev wget \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib && ./configure --prefix=/usr --build=x86_64-unknown-linux-gnu && make && make install \
    && cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install the CPU-only PyTorch build explicitly, before the rest of
# requirements.txt. Render's instances have no GPU, so the default PyPI
# wheel (which bundles the full CUDA toolkit -- cublas, cudnn, cufft,
# triton, etc., ~4.9GB) is both useless and a likely cause of import
# failures (OOM during import, or CUDA initialization errors) on
# memory-constrained instances. Installing this first means
# stable-baselines3's own torch dependency is satisfied by this CPU build
# instead of triggering a second, GPU-enabled install.
RUN pip install --no-cache-dir "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p logs models/weights .runtime \
    && chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
