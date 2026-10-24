#!/usr/bin/env bash
# Single-container entrypoint for Hugging Face Spaces (one public port: $PORT, default 7860).
# FastAPI listens on localhost:8000 (internal only); Streamlit is the public face on $PORT.
set -euo pipefail

PORT="${PORT:-7860}"
export SENTINEL_API_URL="${SENTINEL_API_URL:-http://127.0.0.1:8000}"

uvicorn sentinel.api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT

# Streamlit runs inside an iframe on Spaces: XSRF/CORS protection must be off or file upload returns 403.
exec streamlit run ui/app.py \
  --server.port="$PORT" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableXsrfProtection=false \
  --server.enableCORS=false \
  --browser.gatherUsageStats=false
