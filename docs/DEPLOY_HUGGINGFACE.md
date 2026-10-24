# Deploying Sentinel to Hugging Face (Gradio + ZeroGPU)

On a free Hugging Face account the Gradio SDK with **ZeroGPU** hardware is the practical way to host a Python app.
(Docker Spaces and normal CPU Gradio Spaces are gated behind a paid plan for new accounts; Static Spaces are free
but cannot run Python.) Check the current rules on your account: they change.

## Who can host it
| Account | ZeroGPU hosting |
|---|---|
| Free personal | Up to 2 ZeroGPU Spaces if the account has a verified email and is older than ~30 days (per HF's current guide; older docs say PRO is required, so confirm that **ZeroGPU appears in the hardware list** when you create a Gradio Space). |
| PRO | Up to 10 |

Visitors spend **their own** daily GPU quota (anonymous ~2 min, free account ~5 min, PRO ~40 min), not yours.

## How Sentinel uses ZeroGPU (be realistic)
The only neural parts are the embedder (`bge-small`) and the cross-encoder reranker. They are small, so the GPU is a
modest speed-up, mostly it is the *hosting vehicle*. Design (see `src/sentinel/gpu.py`):

- One module-level `@spaces.GPU(duration=30)` dispatcher exists at import time (ZeroGPU refuses to start without one).
- Models live on CPU; inside the GPU window they move to CUDA, run, and move back.
- Startup ingestion runs on CPU (`gpu.cpu_only()`), so it never needs visitor quota.
- If the GPU call fails for any reason (quota used up, no GPU free) it **falls back to CPU**: a run never fails because
  of the GPU layer. Set `SENTINEL_USE_GPU=0` to disable the GPU path entirely.
- The LLM is *not* on the GPU: it is an API call (Gemini / Groq / Anthropic).

## Steps
1. huggingface.co → **New Space** → SDK **Gradio** → Hardware **ZeroGPU** → Public or Private.
2. Settings → **Variables and secrets**:

   | Mode | Variables | Secrets |
   |---|---|---|
   | **Keyless demo** (safest for public) | none (app auto-falls back to the stand-in LLM when no key is found) | none |
   | **Real LLM** | `SENTINEL_LLM_PROVIDER=gemini`, `SENTINEL_LLM_MODEL=gemini-2.5-flash` | `GOOGLE_API_KEY` |

   Every visitor of a public Space spends **your** LLM-provider quota. Use a free-tier key with a spending/quota cap,
   or stay keyless / private. Keys go in **Secrets**, never in the repo.
3. Push:
   ```bash
   git add -A && git commit -m "HF deploy"        # push_to_hf.sh ships committed files only
   hf auth login
   scripts/push_to_hf.sh <your-hf-username>/sentinel
   ```
   The script prepends the Space's YAML front matter to a temporary copy of the README, so GitHub stays clean.

## Things to check on the first run (not verified in the build environment)
- **torch version**: `requirements.txt` pins `torch==2.8.0`. ZeroGPU only supports certain PyTorch versions: if the build
  complains, check the ZeroGPU docs for the currently supported one and change the pin.
- **Python**: front matter sets `python_version: "3.12"` (ZeroGPU supports 3.10 / 3.12; the project needs ≥ 3.11).
- **GPU context across threads**: LangGraph may run nodes in worker threads. If the Logs tab shows the GPU path never
  being used, that is expected to degrade gracefully to CPU; it will not break answers.
- **Startup**: first boot downloads ~1.4 GB of models and ingests the KB (1-3 min). Later wake-ups reuse the cache.

## Limits of the free setup
- **Ephemeral disk**: the Chroma index and "Past runs" reset on restart; the KB is rebuilt from `data/raw` at startup.
- **Sleep**: idle Spaces pause; the first request after waking is slow.
- **Web search**: DuckDuckGo may rate-limit shared cloud IPs. Set `SENTINEL_SEARCH=none` or add a `TAVILY_API_KEY` secret.
- **Concurrency**: usage counters are process-wide; the Gradio queue serialises runs, which also keeps them accurate.
- **Python sandbox**: best-effort hardening only. Fine for a demo, not a security boundary.
- **Read-only KB**: the Gradio UI has no upload/ingest controls, so visitors cannot modify the knowledge base.

## Troubleshooting
| Symptom | Cause / fix |
|---|---|
| "No @spaces.GPU function detected" | `sentinel.gpu` failed to import or `spaces` is missing: check the build log. |
| Build fails installing torch | Version not supported by ZeroGPU: adjust the pin (see above). |
| ❌ error banner in the UI on run | Missing/invalid API key secret, or a model name the provider retired (`SENTINEL_LLM_MODEL`). |
| Very slow first run | Models are downloading / KB is being built; wait for startup to finish. |
