from google import genai

client = genai.Client()

def embed_text(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return response.embeddings[0].values

print("Gemini client created")

text = "NovaSoft develops software for restaurants."

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text,
)

vector = response.embeddings[0].values

print("Dimensions:", len(vector))
print("First 5 values:", vector[:5])

second_text = "The company builds applications for dining businesses."

second_response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=second_text,
)

second_vector = second_response.embeddings[0].values

print("Second vector dimensions:", len(second_vector))
print("Second vector first 5 values:", second_vector[:5])

dot_product = sum(a * b for a, b in zip(vector, second_vector))

length_a = sum(a * a for a in vector) ** 0.5
length_b = sum(b * b for b in second_vector) ** 0.5

similarity = dot_product / (length_a * length_b)

print(f"Cosine similarity: {similarity:.4f}")

third_text = "A cat is sleeping on the sofa."

third_response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=third_text,
)

third_vector = third_response.embeddings[0].values

third_dot_product = sum(
    a * b for a, b in zip(vector, third_vector)
)

length_c = sum(c * c for c in third_vector) ** 0.5

third_similarity = third_dot_product / (length_a * length_c)

print(f"Original and similar text: {similarity:.4f}")
print(f"Original and cat text: {third_similarity:.4f}")

documents = [second_text, third_text]
scores = [similarity, third_similarity]

best_score = max(scores)
best_index = scores.index(best_score)

print("Best matching document:", documents[best_index])
print("Similarity score:", best_score)

question = input("Ask a question about NovaSoft: ").strip()

question_vector = embed_text(question)

print("\nQuestion:", question)
print("Question vector dimensions:", len(question_vector))

def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    length_a = sum(x * x for x in a) ** 0.5
    length_b = sum(y * y for y in b) ** 0.5

    return dot_product / (length_a * length_b)

ceo_document = "The CEO of NovaSoft is Sarah Miller."

ceo_vector = embed_text(ceo_document)

rag_documents = [text, second_text, third_text, ceo_document]

document_vectors = [vector, second_vector, third_vector, ceo_vector]

rag_scores = [
    cosine_similarity(question_vector, document_vector)
    for document_vector in document_vectors
]

top_k = 2

ranked_indices = sorted(
    range(len(rag_scores)),
    key=lambda index: rag_scores[index],
    reverse=True,
)

print("\n--- Documents ranked for this question ---")
for index in ranked_indices:
    print(f"{rag_scores[index]:.4f} | {rag_documents[index]}")

selected_indices = ranked_indices[:top_k]

context = "\n".join(
    rag_documents[index] for index in selected_indices
)

print("\nQuestion:", question)
print("Retrieved context:", context)

prompt = f"""
Answer the question using only the information in the context.
If the context does not contain the answer, say:
"I don't know based on the provided context."

Context:
{context}

Question:
{question}
"""

print("\nPrompt sent to Gemini:")
print(prompt)

answer_response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt,
)

print("\nGemini's answer:")
print(answer_response.text)