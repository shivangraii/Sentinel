FROM python:3.11-slim

ARG EXTRAS="gemini,groq,ollama,anthropic,search,ui,embeddings"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    HF_HOME=/app/storage/hf

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[${EXTRAS}]"

COPY configs ./configs
COPY data ./data
COPY ui ./ui
COPY eval ./eval
COPY scripts ./scripts

# run as an unprivileged user; storage/ holds Chroma, checkpoints and cached models
RUN useradd -m sentinel && mkdir -p /app/storage /app/data/reports /app/traces && chown -R sentinel /app
USER sentinel

EXPOSE 8000 8501
CMD ["uvicorn", "sentinel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
