def retrieve_chunks(query, model, index, chunks, top_k=3):
    import numpy as np

    # Convert query to embedding
    query_embedding = model.encode([query])

    # Search FAISS
    distances, indices = index.search(np.array(query_embedding), top_k)

    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return results