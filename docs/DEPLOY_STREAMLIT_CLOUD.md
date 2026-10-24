# Deploying Sentinel on Streamlit Community Cloud (free)

Use this when Hugging Face ZeroGPU is not available to your account. It is a different platform from Hugging Face,
free for public apps, and deploys straight from your GitHub repo. Optionally pair it with a free **Static** Hugging Face
Space as a landing page (step 5).

## What runs
`deploy/streamlit/streamlit_app.py` calls the research graph **in-process** (Community Cloud cannot run a separate API
server). To fit the free tier's small RAM it uses the **light profile**: hash embeddings + lexical reranker (no torch).
The planner, researcher, analyst (SQL), writer, critic and human-approval flow are unchanged; only retrieval quality is
lower than with the bge models. The default LLM is Gemini via API key; with no key it falls back to the keyless stand-in.

## Steps
1. **Push the repo to GitHub** (public, and you need admin rights on it). Your current remote points to a repo named
   `Sage`: rename the repo or run `git remote set-url origin https://github.com/<you>/Sentinel.git`.
2. Go to **share.streamlit.io**, sign in with GitHub → **Create app** → *Deploy a public app from GitHub*:
   - Repository / branch: your repo, `main` (or `master`)
   - **Main file path: `deploy/streamlit/streamlit_app.py`**
3. **Advanced settings**: Python **3.12** (3.11 also works). Paste your secrets (see
   `deploy/streamlit/secrets.toml.example`): at minimum `GOOGLE_API_KEY`. Never commit a real secrets file.
4. **Deploy.** First boot installs dependencies and builds the knowledge base (a few minutes).
5. *(Optional)* **Hugging Face Static Space** as a landing page: create a Space with SDK **Static**, upload
   `deploy/hf_static/index.html` and `deploy/hf_static/README.md` (Files → Add file), after replacing both
   `YOUR-APP.streamlit.app` placeholders in `index.html`.

### Why the dependency file lives in `deploy/streamlit/`
Community Cloud looks for the dependency file in the entrypoint's folder first, then the repo root. The root
`requirements.txt` is the heavy torch set for the Hugging Face Space, so the slim one sits next to the entrypoint and
wins. Only one dependency file is used, so don't add a `pyproject`/`Pipfile` into `deploy/streamlit/`.

## Protecting your API key
Every visitor of a public app spends your LLM quota. Built-in protection: a demo-wide cap of **30 research runs per
hour** (`SENTINEL_MAX_RUNS_PER_HOUR`, `0` = unlimited), runs are serialised, and there are no upload/ingest controls.
Also use a free-tier key with a quota cap in the provider console. Lower `SENTINEL_MAX_LOOPS` to cut calls per run.

## Limits and expectations
- **Resources**: shared and small (roughly 0.7-2.7 GB RAM per Streamlit's own figures). The light profile is intended to
  fit; I have not measured Sentinel's actual footprint on Community Cloud.
- **Sleep**: apps hibernate after ~12 h without traffic; a visitor can wake them.
- **Ephemeral disk**: the knowledge base is rebuilt on every start; "Past runs" resets.
- **Web search**: DuckDuckGo may rate-limit shared IPs. Set `SENTINEL_SEARCH = "none"` if you see failures.
- **Python sandbox**: best-effort hardening, not a security boundary (fine for a demo).

## Run it locally first
```bash
pip install -e ".[dev,ui,gemini]"
streamlit run deploy/streamlit/streamlit_app.py
```

## Troubleshooting
| Symptom | Cause / fix |
|---|---|
| Build fails on `chromadb` / "sqlite3 >= 3.35" | `pysqlite3-binary` missing from `deploy/streamlit/requirements.txt`, or the root requirements file was picked up instead. |
| "Oh no. Error running app" with `ModuleNotFoundError` | Dependency file not found: check the main file path is exactly `deploy/streamlit/streamlit_app.py`. |
| App killed / restarts during a run | Out of memory: remove unused provider packages from `deploy/streamlit/requirements.txt`. |
| "rate-limited" message | The hourly demo cap was reached; raise `SENTINEL_MAX_RUNS_PER_HOUR` or wait. |
| Red error `RuntimeError: Provider ... needs extra packages` | Add that provider's `langchain-*` package to `deploy/streamlit/requirements.txt`. |
