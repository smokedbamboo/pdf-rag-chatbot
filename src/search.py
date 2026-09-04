import pickle

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def search(query):
    with open("indexes/index.pkl", "rb") as f:
        index = pickle.load(f)

    chunks = index["chunks"]
    embeddings = index["embeddings"]

    query_embedding = model.encode(query)

    best_score = -1
    best_idx = 0

    for i, embedding in enumerate(embeddings):
        score = cos_sim(query_embedding, embedding).item()

        if score > best_score:
            best_score = score
            best_idx = i

    return chunks[best_idx], best_score


if __name__ == "__main__":
    query = input("Ask a question: ")

    chunk, score = search(query)

    print("\nBest Match:")
    print("-" * 50)
    print(chunk)

    print(f"\nSimilarity Score: {score:.4f}")