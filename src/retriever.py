import faiss
import pickle

from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def retrieve(query, chunks, index, k=3):
    query_embedding = model.encode([query]).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for idx, distance in zip(indices[0], distances[0]):
        results.append(
            (chunks[idx], distance)
        )

    return results


if __name__ == "__main__":
    index = faiss.read_index(
        "indexes/faiss.index"
    )

    with open("indexes/metadata.pkl", "rb") as f:
        chunks = pickle.load(f)

    query = "What is self attention?"

    results = retrieve(
        query,
        chunks,
        index,
        k=3
    )

    for chunk, distance in results:
        print(f"{distance:.4f}")
        print(chunk["text"])
        print("-" * 50)