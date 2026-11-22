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