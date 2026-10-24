# One image, two ways to run it:
#   * Hugging Face Spaces / `docker run -p 7860:7860 ...`  -> default CMD (API + UI in one container)
#   * docker compose                                        -> compose overrides `command` per service
FROM python:3.11-slim

# Provider extras you actually use. Drop `embeddings` for a tiny image (then set SENTINEL_EMBEDDINGS=hash).
ARG EXTRAS="gemini,groq,anthropic,search,ui,embeddings"
ARG EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
ARG RERANKER_MODEL="BAAI/bge-reranker-base"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    HF_HOME=/app/hf-cache \
    PORT=7860

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[${EXTRAS}]"

# Bake the models into the image: Space disks are ephemeral, so downloading at runtime would
# re-fetch ~1.4 GB on every restart and make the first request take minutes.
RUN if echo "${EXTRAS}" | grep -q embeddings; then \
      python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBEDDING_MODEL}'); CrossEncoder('${RERANKER_MODEL}')"; \
    fi

COPY configs ./configs
COPY data ./data
COPY ui ./ui
COPY eval ./eval
COPY scripts ./scripts
COPY start.sh ./start.sh

# Spaces runs containers as uid 1000; make that explicit. storage/ holds Chroma + checkpoints.
RUN useradd -m -u 1000 sentinel \
    && mkdir -p /app/storage /app/data/reports /app/traces \
    && sed -i 's/\r$//' /app/start.sh \
    && chmod +x /app/start.sh \
    && chown -R sentinel:sentinel /app
USER sentinel

EXPOSE 7860
CMD ["./start.sh"]
