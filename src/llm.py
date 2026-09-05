from ollama import chat


def ask_llm(prompt):
    response = chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":
    question = input("Ask: ")

    answer = ask_llm(question)

    print("\nAnswer:")
    print(answer)