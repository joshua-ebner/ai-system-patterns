.PHONY: rag agent down build

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
