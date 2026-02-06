import json
import shutil
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ---------- Paths ----------
CHUNKS_FILE = Path("data/processed/langchain/chunks.jsonl")
DB_DIR = Path("data/vectorstore/langchain")

# ---------- Rebuild DB cleanly (dev-friendly) ----------
if DB_DIR.exists():
    shutil.rmtree(DB_DIR)

DB_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Load chunks ----------
docs = []

with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
    for line in file:
        row = json.loads(line)
        docs.append(
            Document(
                page_content=row["text"],   
                metadata=row["metadata"]
            )
        )

print(f"Loaded {len(docs)} chunks")
print(f"Embedding {len(docs)} documents...")

# ---------- Embed ----------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    chunk_size=100  # helps avoid rate limits on larger corpora
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=str(DB_DIR)
)

print(f"Vectorstore saved → {DB_DIR}")
print("Done.")
