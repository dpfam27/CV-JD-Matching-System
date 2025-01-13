from pathlib import Path
import fitz
pdf_dir = Path(r"C:\Users\pngoc\OneDrive\Desktop\VNU IS\AI Project\CV-JD-Matching-System\data\raw\cv\test")
print(pdf_dir.exists())
if pdf_dir.exists():
    pdfs = list(pdf_dir.glob("*.pdf"))
    print(f"Found PDFs: {[p.name for p in pdfs]}")