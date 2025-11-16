.PHONY: dev
dev:
	source .venv/bin/activate && uvicorn backend.app:app --reload

.PHONY: lint
lint:
	flake8 . && black --check . && isort --check-only .

.PHONY: format
format:
	autoflake --remove-all-unused-imports --in-place --recursive . && black . && isort .

.PHONY: test
test:
	pytest