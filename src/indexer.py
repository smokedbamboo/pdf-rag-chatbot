import pickle
import faiss

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

    print("Embedding shape:", embeddings.shape)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings.astype("float32"))

    print("FAISS index size:", index.ntotal)

    with open("indexes/metadata.pkl", "wb") as f:
        pickle.dump(chunk_records, f)

    faiss.write_index(
        index,
        "indexes/faiss.index"
    )

    print(f"Saved {len(chunks)} chunks")


if __name__ == "__main__":
    build_index("data/Attention_Is_All_You_Need.pdf")