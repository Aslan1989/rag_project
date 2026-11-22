from pathlib import Path

from google import genai


client = genai.Client()


def embed_text(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return response.embeddings[0].values


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    length_a = sum(x * x for x in a) ** 0.5
    length_b = sum(y * y for y in b) ** 0.5
    return dot_product / (length_a * length_b)


file_path = Path(__file__).with_name("novasoft.txt")
text = file_path.read_text(encoding="utf-8").strip()

documents = [
    text,
    "The company builds applications for dining businesses.",
    "A cat is sleeping on the sofa.",
    "The CEO of NovaSoft is Sarah Miller.",
]

question = input("Ask a question about NovaSoft: ").strip()
question_vector = embed_text(question)

document_vectors = [embed_text(document) for document in documents]
scores = [
    cosine_similarity(question_vector, document_vector)
    for document_vector in document_vectors
]

top_k = 2
ranked_indices = sorted(
    range(len(scores)),
    key=lambda index: scores[index],
    reverse=True,
)

# for index in ranked_indices:
#     print(f"{scores[index]:.4f} | {documents[index]}")

selected_indices = ranked_indices[:top_k]
context = "\n".join(documents[index] for index in selected_indices)

print("\nRetrieved context:")
print(context)

prompt = f"""
Answer the question using only the information in the context.
If the context does not contain the answer, say:
"I don't know based on the provided context."

Context:
{context}

Question:
{question}
"""

answer_response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt,
)

print("\nGemini's answer:")
print(answer_response.text)