# =====================================================================
# Dockerfile do medica-ia-multimodal.
# Imagem base alinhada ao requires-python do pyproject.toml (>=3.12).
# Otimizado para Hugging Face Spaces (mesma imagem) e desenvolvimento local.
# =====================================================================

FROM python:3.12-slim AS base

# Variaveis de ambiente padrao para Python em container.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Dependencias de sistema necessarias para libs como opencv, librosa e ffmpeg.
# Mantido enxuto: cada nova lib pesada deve justificar entrada aqui.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia primeiro o requirements para aproveitar cache do Docker em builds incrementais.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia o restante do projeto. O indice Chroma (data/processed/chroma/)
# vem versionado via LFS pra evitar timeout de build no HF Spaces:
# gerar embeddings bge-m3 em CPU leva ~30min sem GPU, alem do timeout
# do hardware free. Regerar local com `scripts/build_rag_index.py` ao
# trocar PDFs ou embedding model.
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
