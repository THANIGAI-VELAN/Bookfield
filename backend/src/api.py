from fastapi import FastAPI
from pydantic import BaseModel

from src.chunk_text import chunk_text
from src.embeddings import create_embeddings, model
from src.vector_store import build_index
from src.retriever import retrieve_chunks
from src.llm import generate_answer


app = FastAPI()


# Request format
class QueryRequest(BaseModel):
    question: str
    book: str


# Load books once
books = {
    "art_of_war": "data/art_of_war.txt",
    "meditations": "data/meditations.txt"
}

indexes = {}
chunks_store = {}


def load_book(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Preload all books (IMPORTANT)
for book_name, path in books.items():
    text = load_book(path)

    raw_chunks = chunk_text(text)

    chunks = []
    for i, ch in enumerate(raw_chunks):
        chunks.append({
            "text": ch,
            "id": i
        })

    texts = [c["text"] for c in chunks]

    embeddings = create_embeddings(texts)
    index = build_index(embeddings)

    indexes[book_name] = index
    chunks_store[book_name] = chunks


@app.post("/ask")
def ask_question(request: QueryRequest):
    query = request.question
    book = request.book

    index = indexes[book]
    chunks = chunks_store[book]

    results = retrieve_chunks(query, model, index, chunks)

    context = "\n\n".join([r["text"] for r in results])

    answer = generate_answer(query, context)

    sources = [r["id"] for r in results]

    return {
        "answer": answer,
        "sources": sources
    }