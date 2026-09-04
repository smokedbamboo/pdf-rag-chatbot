from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


if __name__ == "__main__":
    sentence1 = "Transformers use self attention"
    sentence2 = "Attention mechanisms are used in transformers"
    sentence3 = "Cats are mammals"

    emb1 = model.encode(sentence1)
    emb2 = model.encode(sentence2)
    emb3 = model.encode(sentence3)

    sim12 = cos_sim(emb1, emb2)
    sim13 = cos_sim(emb1, emb3)

    print(f"Similarity(1,2): {sim12.item():.4f}")
    print(f"Similarity(1,3): {sim13.item():.4f}")