# from pathlib import Path

from google import genai

import chromadb

client = genai.Client()


def embed_text(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return response.embeddings[0].values


# def cosine_similarity(a, b):
#     dot_product = sum(x * y for x, y in zip(a, b))
#     length_a = sum(x * x for x in a) ** 0.5
#     length_b = sum(y * y for y in b) ** 0.5
#     return dot_product / (length_a * length_b)


# file_path = Path(__file__).with_name("novasoft.txt")
# file_text = file_path.read_text(encoding="utf-8").strip()

# documents = file_text.split("\n\n")
# print("Chunks loaded:", len(documents))

# question = input("Ask a question about NovaSoft: ").strip()
# question_vector = embed_text(question)

# document_vectors = [embed_text(document) for document in documents]

# chroma_client = chromadb.PersistentClient(path="./chroma_data")
# collection = chroma_client.get_or_create_collection(
#     name="novasoft",
#     embedding_function=None,
# )

# collection.upsert(
#     ids=[f"chunk-{index}" for index in range(len(documents))],
#     documents=documents,
#     embeddings=document_vectors,
# )

# print("Chunks saved in Chroma:", collection.count())

# scores = [
#     cosine_similarity(question_vector, document_vector)
#     for document_vector in document_vectors
# ]

# top_k = 2
# ranked_indices = sorted(
#     range(len(scores)),
#     key=lambda index: scores[index],
#     reverse=True,
# )

# # for index in ranked_indices:
# #     print(f"{scores[index]:.4f} | {documents[index]}")

# selected_indices = ranked_indices[:top_k]
# context = "\n".join(documents[index] for index in selected_indices)

chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_collection(
    name="novasoft_pdf_chunks",
    embedding_function=None,
)

question = input("Ask a question about NovaSoft: ").strip()
question_vector = embed_text(question)

results = collection.query(
    query_embeddings=[question_vector],
    n_results=1,
    include=["documents", "metadatas"],
)

matched_chunks = results["documents"][0]
matched_metadata = results["metadatas"][0]

if matched_metadata:
    first = matched_metadata[0]
    next_number = first["chunk"] + 1
    next_id = f"{first['source']}-p{first['page']}-c{next_number}"

    if next_id not in results["ids"][0]:
        neighbor = collection.get(ids=[next_id], include=["documents", "metadatas"])
        if neighbor["ids"]:
            matched_chunks.insert(1, neighbor["documents"][0])
            matched_metadata.insert(1, neighbor["metadatas"][0])

context = "\n".join(
    f"[{meta['source']}, page {meta['page']}, chunk {meta['chunk']}] {chunk}"
    for chunk, meta in zip(matched_chunks, matched_metadata)
)
print("\nRetrieved context:")
print(context)

prompt = f"""
Answer the question using only the information in the context.
Cite each factual claim using the source label exactly as it appears in the context.
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