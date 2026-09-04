import pickle

from pdf_loader import load_pdf
from chunker import chunk_text
from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def build_index(pdf_path):
    pages = load_pdf(pdf_path)

    full_text = " ".join(pages)

    chunks = chunk_text(full_text)

    embeddings = model.encode(chunks)

    index = {
        "chunks": chunks,
        "embeddings": embeddings
    }

    with open("indexes/index.pkl", "wb") as f:
        pickle.dump(index, f)

    print(f"Saved {len(chunks)} chunks")


if __name__ == "__main__":
    build_index("data/Attention_Is_All_You_Need.pdf")