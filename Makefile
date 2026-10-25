PYTHON      ?= .venv/bin/python
UV          ?= uv
BACKEND_DIR  = backend
FRONTEND_DIR = frontend
BACKEND_PORT ?= 8000
FRONTEND_PORT ?= 5173

.PHONY: help setup install-backend install-frontend \
	backend frontend run \
	health build typecheck test clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install all dependencies (backend + frontend)
	$(UV) venv .venv
	$(UV) pip install --python .venv/bin/python -e . 2>/dev/null || true
	@$(MAKE) install-backend
	@$(MAKE) install-frontend

install-backend: ## Install Python deps into .venv (CPU torch)
	@[ -d .venv ] || $(UV) venv .venv
	cd backend && ../.venv/bin/python -m pip install -r requirements.txt

install-frontend: ## Install frontend node deps
	cd frontend && npm install

backend: ## Run the FastAPI backend on :$(BACKEND_PORT)
	cd $(BACKEND_DIR) && PYTHONPATH=. ../$(PYTHON) -m uvicorn app.main:app \
		--host 127.0.0.1 --port $(BACKEND_PORT) --reload

frontend: ## Run the Vite dev server
	cd $(FRONTEND_DIR) && npm run dev

run: ## Run backend and frontend together (backend:8000, frontend:5173)
	@echo "Starting backend on http://127.0.0.1:$(BACKEND_PORT) and frontend on http://localhost:$(FRONTEND_PORT)"
	@trap 'kill 0' INT TERM EXIT; \
	( cd $(BACKEND_DIR) && PYTHONPATH=. ../$(PYTHON) -m uvicorn app.main:app \
		--host 127.0.0.1 --port $(BACKEND_PORT) ) & \
	( cd $(FRONTEND_DIR) && npm run dev ) & \
	wait

build: ## Production build of the frontend
	cd $(FRONTEND_DIR) && npm run build

typecheck: ## Type-check the frontend (tsc)
	cd $(FRONTEND_DIR) && npx tsc -b

test: ## Run backend tests
	cd $(BACKEND_DIR) && PYTHONPATH=. ../$(PYTHON) -m pytest -q

clean: ## Remove build artifacts, node_modules and __pycache__
	rm -rf $(FRONTEND_DIR)/dist $(FRONTEND_DIR)/node_modules
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
