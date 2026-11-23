from pathlib import Path

from pypdf import PdfReader

pdf_path = Path(__file__).with_name("sample.pdf")
reader = PdfReader(pdf_path)

print("Total pages:", len(reader.pages))

page_records = []

for page_number, page in enumerate(reader.pages, start=1):
    text = (page.extract_text() or "").strip()
    if not text:
        continue

    page_records.append({
        "source": pdf_path.name,
        "page": page_number,
        "text": text,
    })

gemini = genai.Client()
chroma = chromadb.PersistentClient(path="./chroma_data")
collection = chroma.get_or_create_collection(
    name="novasoft_pdf",
    embedding_function=None,
)

for record in page_records:
    response = gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=record["text"],
    )

    collection.upsert(
        ids=[f"{record['source']}-page-{record['page']}"],
        documents=[record["text"]],
        embeddings=[response.embeddings[0].values],
        metadatas=[{"source": record["source"], "page": record["page"]}],
    )

print("PDF pages saved in Chroma:", collection.count())