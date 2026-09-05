import pickle

from retriever import retrieve
from prompt_builder import build_prompt
from llm import ask_llm


def load_index():
    with open("indexes/index.pkl", "rb") as f:
        return pickle.load(f)


def main():
    index = load_index()

    chunks = index["chunks"]
    embeddings = index["embeddings"]

    while True:
        question = input("\nAsk a question (type 'exit' to quit): ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        results = retrieve(
            question,
            chunks,
            embeddings,
            k=3
        )

        print("\nRetrieved Chunks:")
        print("=" * 80)

        for i, (chunk_record, score) in enumerate(results, start=1):
            print(f"\n[{i}] Score: {score:.4f}")
            print("-" * 80)

            print(f"Source: {chunk_record['source']}")
            print()

            print(chunk_record["text"])

        print("\n" + "=" * 80)

        retrieved_chunks = [
            chunk_record["text"]
            for chunk, score in results
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