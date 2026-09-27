PYTHON ?= python3

.PHONY: test check api worker up down smoke smoke-asr probe

test:
	PYTHONPATH=packages/audio-core/src:apps/platform-api/src:apps/ingest-worker/src:apps/asr-eval/src:apps/vad-eval/src:apps/diarization-eval/src $(PYTHON) -m unittest discover -s tests -v

check:
	PYTHONPATH=packages/audio-core/src:apps/platform-api/src:apps/ingest-worker/src:apps/asr-eval/src:apps/vad-eval/src:apps/diarization-eval/src $(PYTHON) -m compileall -q packages apps tests

api:
	PYTHONPATH=packages/audio-core/src:apps/platform-api/src $(PYTHON) -m platform_api.main

worker:
	PYTHONPATH=packages/audio-core/src:apps/platform-api/src:apps/ingest-worker/src $(PYTHON) -m ingest_worker.main

up:
	docker compose -f infra/compose.yaml up --build

down:
	docker compose -f infra/compose.yaml down

smoke:
	$(PYTHON) scripts/smoke_stack.py

smoke-asr:
	$(PYTHON) scripts/smoke_stack.py --transcribe

probe:
	PYTHONPATH=packages/audio-core/src $(PYTHON) -m audio_core.cli probe $(FILE)
