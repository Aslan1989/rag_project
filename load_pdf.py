from pathlib import Path

from pypdf import PdfReader

pdf_path = Path(__file__).with_name("sample.pdf")
reader = PdfReader(pdf_path)

print("Total pages:", len(reader.pages))

first_page = reader.pages[0]
text = first_page.extract_text()

print("Page 1 preview:")
print(text[:500])