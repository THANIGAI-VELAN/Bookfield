
def chunk_text(text, chunk_size=80, overlap=20):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        start = start + chunk_size - overlap

    return chunks