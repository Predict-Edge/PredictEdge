.PHONY: dev
dev:
	source .venv/bin/activate && uvicorn backend.app:app --reload

.PHONY: lint
lint:
	flake8 . && black --check . && isort --check-only .
	# flake8 . --exclude='*.ipynb' && black --check . --exclude='\.ipynb$' && isort --check-only . --skip-glob='*.ipynb'

.PHONY: format
format:
	autoflake --remove-all-unused-imports --in-place --recursive . && black . && isort .
	# autoflake --remove-all-unused-imports --in-place --recursive . --exclude='*.ipynb' && black . --exclude='\.ipynb$' && isort . --skip-glob='*.ipynb'

.PHONY: test
test:
	pytest