# src/read_books.py

from chunk_text import chunk_text
from embeddings import create_embeddings, model
from vector_store import build_index
from retriever import retrieve_chunks
from llm import generate_answer


def load_book(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


if __name__ == "__main__":

    # 🔹 Step 1: Define available books
    books = {
        "art_of_war": "data/art_of_war.txt",
        "meditations": "data/meditations.txt"
    }

    # 🔹 Step 2: Show books to user
    print("Available books:")
    for b in books:
        print("-", b)

    # 🔹 Step 3: Select book
    selected_book = input("Select a book: ").strip()

    # 🔹 Basic validation
    if selected_book not in books:
        print("Invalid book selection!")
        exit()

    # 🔹 Step 4: Load selected book
    book_path = books[selected_book]
    book_text = load_book(book_path)

    # 🔹 Step 5: Chunk the text
    chunks = chunk_text(book_text)

    print("Total chunks:", len(chunks))

    # 🔹 Step 6: Create embeddings
    texts = [c["text"] for c in chunks]
    embeddings = create_embeddings(texts)

    # 🔹 Step 7: Build FAISS index
    index = build_index(embeddings)

    print("\nSystem ready! Ask questions.\n")

    # 🔹 Step 8: Question loop
    while True:
        query = input("Ask a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        # 🔹 Step 9: Retrieve relevant chunks
        results = retrieve_chunks(query, model, index, chunks)

        # 🔹 Combine chunks into context
        context = "\n\n".join(r["text"] for r in results)

        # 🔹 Step 10: Generate answer
        answer = generate_answer(query, context)

        print("\nAnswer:\n")
        print(answer)
        print("\n" + "-"*50 + "\n")
        print("\n Sources: \n")
        for r in results:
            print(f"- {r['text']}")