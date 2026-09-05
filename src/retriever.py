from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def retrieve(query, chunks, k=3):
    query_embedding = model.encode(query)

    chunk_embeddings = model.encode(chunks)

    scores = []

    for chunk, chunk_embedding in zip(chunks, chunk_embeddings):
        score = cos_sim(query_embedding, chunk_embedding).item()

        scores.append((chunk, score))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    return scores[:k]


if __name__ == "__main__":
    chunks = [
        "Transformers use self attention",
        "Cats are mammals",
        "Positional encoding adds order information"
    ]

    query = "What is self attention?"

    results = retrieve(query, chunks, k=2)

    for chunk, score in results:
        print(f"Score: {score:.4f}")
        print(chunk)
        print("-" * 50)