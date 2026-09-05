from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def retrieve(query, chunks, embeddings, k=3):
    query_embedding = model.encode(query)

    scores = []

    for chunk, embedding in zip(chunks, embeddings):
        score = cos_sim(query_embedding, embedding).item()

        scores.append((chunk, score))

    scores.sort(
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

    embeddings = model.encode(chunks)

    query = "What is self attention?"

    results = retrieve(
        query,
        chunks,
        embeddings,
        k=2
    )

    for chunk, score in results:
        print(f"{score:.4f} | {chunk}")