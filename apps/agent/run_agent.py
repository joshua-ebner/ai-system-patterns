from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from .agent_graph import agent

import json
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path("logs/agent_runs_v1.jsonl")

def log_agent_run(payload: dict):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def main():
    print("\nAgent ready. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in ["exit", "quit"]:
            break

        # ---- timing start ----
        start = time.time()

        try:
            result = agent.invoke({
                "messages": [HumanMessage(content=query)],
                "rag_result": None,
                "retry_count": 0,}
            )
            answer = result["messages"][-1].content
        except Exception as e:
            answer = f"Agent error: {str(e)}"

        latency = time.time() - start

        print("\nAssistant:", answer, "\n")

        refused = "don't have enough relevant context" in answer.lower() \
                   or "not enough relevant context" in answer.lower()

        error = answer.startswith("Agent error:")


        log_agent_run({
            "ts": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "answer": answer,
            "refused": refused,
            "error": error,
            "latency_sec": latency,
        })


if __name__ == "__main__":
    main()
