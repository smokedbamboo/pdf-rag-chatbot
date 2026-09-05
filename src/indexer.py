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
    chunk_records = [
        {
            "text": chunk,
            "source": pdf_path
        }
        for chunk in chunks
    ]

    embeddings = model.encode(chunks)

    index = {
        "chunks": chunk_records,
        "embeddings": embeddings
    }

    with open("indexes/index.pkl", "wb") as f:
        pickle.dump(index, f)

    print(f"Saved {len(chunks)} chunks")


if __name__ == "__main__":
    build_index("data/Attention_Is_All_You_Need.pdf")