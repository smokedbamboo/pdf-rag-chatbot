import faiss
import pickle

from retriever import retrieve
from prompt_builder import build_prompt
from llm import ask_llm


def load_index():
    index = faiss.read_index(
        "indexes/faiss.index"
    )

    with open("indexes/metadata.pkl", "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def main():
    index, chunks = load_index()

    while True:
        question = input("\nAsk a question (type 'exit' to quit): ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        results = retrieve(
            question,
            chunks,
            index,
            k=3
        )

        print("\nRetrieved Chunks:")
        print("=" * 80)

        for i, (chunk_record, distance) in enumerate(results, start=1):
            print(f"\n[{i}] Distance: {distance:.4f}")
            print("-" * 80)

            print(f"Source: {chunk_record['source']}")
            print()

            print(chunk_record["text"])

        print("\n" + "=" * 80)

        retrieved_chunks = [
            chunk_record["text"]
            for chunk_record, distance in results
        ]

        prompt = build_prompt(
            question,
            retrieved_chunks
        )

        answer = ask_llm(prompt)

        print("\nAnswer:")
        print("-" * 80)
        print(answer)


if __name__ == "__main__":
    main()