from pathlib import Path

import chromadb
from google import genai

file_path = Path(__file__).with_name("novasoft.txt")
file_text = file_path.read_text(encoding="utf-8")

chunks = [
    paragraph.strip()
    for paragraph in file_text.split("\n\n")
    if paragraph.strip()
]

print("Chunks ready to index:", len(chunks))

gemini = genai.Client()
vectors = []

for chunk in chunks:
    response = gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk,
    )
    vectors.append(response.embeddings[0].values)

chroma = chromadb.PersistentClient(path="./chroma_data")
collection = chroma.get_or_create_collection(
    name="novasoft",
    embedding_function=None,
)

collection.upsert(
    ids=[f"chunk-{index}" for index in range(len(chunks))],
    documents=chunks,
    embeddings=vectors,
)

print("Chunks saved in Chroma:", collection.count())