.PHONY: rag agent down build mcp-local rag-local up logs ps restart

build:
	docker compose build

rag:
	docker compose up -d rag-api

agent:
	-docker run --rm -it \
		--network ai-engineering-patterns_default \
		--env-file .env \
		-e RAG_API_URL=http://rag-api:8000/query \
		agent:dev

down:
	docker compose down

up:
	docker compose up --build

logs:
	docker compose logs -f
	
ps:
	docker compose ps

restart:
	docker compose down
	docker compose up --build

# --- Local dev (non-docker) ---
mcp-local:
	uvicorn apps.mcp.mcp_api:app --reload --port 8001

rag-local:
	uvicorn apps.rag.rag_api:app --reload --port 8000