from google import genai

client = genai.Client()

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