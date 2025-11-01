.PHONY: dev

dev:
	source .venv/bin/activate && uvicorn backend.app:app --reload
