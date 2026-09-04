def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()

    chunks = []

    stride = chunk_size - overlap

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk_words = words[start:end]

        chunk = " ".join(chunk_words)

        chunks.append(chunk)

        start += stride

    return chunks
if __name__ == "__main__":
    text = " ".join(str(i) for i in range(1, 21))

    chunks = chunk_text(text, chunk_size=5, overlap=2)

    for i, chunk in enumerate(chunks, start=1):
        print(f"Chunk {i}: {chunk}")