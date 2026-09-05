def build_prompt(question, chunks):
    context = "\n\n".join(chunks)

    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the provided context.

If the answer is not present in the context, say:
"I don't know based on the provided document."

Context:
---------
{context}
---------

Question:
{question}

Answer:
"""

    return prompt