.PHONY: install install-all demo test lint ingest ask serve ui eval eval-all docker db golden

install:        ## core + dev deps (enough for the offline demo and the tests)
	pip install -e ".[dev]"

install-all:    ## everything: LLM providers, embeddings, search, UI
	pip install -e ".[all]"

demo:           ## offline end-to-end run, no API key needed
	SENTINEL_OFFLINE=1 sentinel demo

test:
	pytest

lint:
	ruff check .

ingest:
	sentinel ingest --reset

ask:            ## make ask Q="your question"
	sentinel ask "$(Q)" --trace

serve:
	sentinel serve

ui:
	streamlit run ui/app.py

eval:           ## retrieval ablation (no LLM needed)
	sentinel eval --mode retrieval

eval-all:       ## retrieval + end-to-end (uses your configured LLM)
	sentinel eval --mode all

db:             ## regenerate the synthetic missions database
	python scripts/make_sample_db.py

golden:         ## rebuild the golden evaluation set
	python scripts/build_golden_set.py

docker:
	docker compose up --build
