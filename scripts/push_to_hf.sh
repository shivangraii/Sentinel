#!/usr/bin/env bash
# Publish the current commit to a Hugging Face *Gradio* Space (ZeroGPU) without touching your GitHub README.
# HF reads its config from YAML front matter at the top of README.md; GitHub would render that as an ugly table,
# so it is added only to a throwaway copy.
#
# usage: scripts/push_to_hf.sh <hf-username>/<space-name>
# auth : `hf auth login` (or `huggingface-cli login`) once, with a token that has write access.
# note : create the Space first in the web UI (SDK: Gradio, hardware: ZeroGPU). Hardware is chosen there, not here.
set -euo pipefail
SPACE="${1:?usage: $0 <hf-username>/<space-name>}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git archive HEAD | tar -x -C "$TMP"      # committed files only: no .env, no storage/, no .git history
rm -rf "$TMP/.github" "$TMP/Dockerfile" "$TMP/docker-compose.yml" "$TMP/start.sh" "$TMP/deploy"   # not used by a Gradio Space
{
  cat <<'FM'
---
title: Sentinel
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
python_version: "3.12"
pinned: false
short_description: Self-verifying multi-agent deep-research platform
---

FM
  cat "$TMP/README.md"
} > "$TMP/README.hf" && mv "$TMP/README.hf" "$TMP/README.md"

cd "$TMP"
git init -q -b main
git add -A
git -c user.name="deploy" -c user.email="deploy@local" commit -q -m "Deploy $(date -u +%FT%TZ)"
git push --force "https://huggingface.co/spaces/${SPACE}" main
echo "Pushed. Build logs: https://huggingface.co/spaces/${SPACE}"
