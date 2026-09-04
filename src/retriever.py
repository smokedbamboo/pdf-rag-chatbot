from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def retrieve(query, chunks):
    query_embedding = model.encode(query)

    chunk_embeddings = model.encode(chunks)

    best_score = -1
    best_idx = 0

    for i, chunk_embedding in enumerate(chunk_embeddings):
        score = cos_sim(query_embedding, chunk_embedding).item()

        if score > best_score:
            best_score = score
            best_idx = i

    return chunks[best_idx]


if __name__ == "__main__":
    chunks = [
        "Transformers use self attention",
        "Cats are mammals",
        "Positional encoding adds order information"
    ]

    query = "What is self attention?"

    result = retrieve(query, chunks)

    print(result)