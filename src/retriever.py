import chromadb


client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    name="pdf_rag"
)


def retrieve(query, k=3):
    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    retrieved = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        retrieved.append(
            (
                {
                    "text": document,
                    "source": metadata["source"]
                },
                distance
            )
        )

    return retrieved