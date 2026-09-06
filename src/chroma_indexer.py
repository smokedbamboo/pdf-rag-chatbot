import chromadb

from pdf_loader import load_pdf
from chunker import chunk_text


def build_collection(pdf_path):
    client = chromadb.PersistentClient(path="chroma_db")

    collection = client.get_or_create_collection(
        name="pdf_rag"
    )

    pages = load_pdf(pdf_path)

    full_text = " ".join(pages)

    chunks = chunk_text(full_text)

    collection.add(
        documents=chunks,
        metadatas=[
            {"source": pdf_path}
            for _ in chunks
        ],
        ids=[
            f"chunk_{i}"
            for i in range(len(chunks))
        ]
    )

    print(f"Added {len(chunks)} chunks")


if __name__ == "__main__":
    build_collection(
        "data/Attention_Is_All_You_Need.pdf"
    )