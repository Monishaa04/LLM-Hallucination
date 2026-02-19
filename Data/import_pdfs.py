import os
import csv
import fitz  # PyMuPDF
import re
from tqdm import tqdm

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data', 'raw_papers')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data', 'ml_texts')
META_FILE = os.path.join(os.path.dirname(__file__), '..', 'Data', 'metadata.csv')

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

CLEAN_RE = re.compile(r"\s+", re.UNICODE)


def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    parts = []
    for page in doc:
        text = page.get_text()
        if text:
            parts.append(text)
    doc.close()
    return "\n".join(parts)


def clean_text(text: str) -> str:
    text = text.replace('\x0c', '\n')
    text = CLEAN_RE.sub(' ', text)
    return text.strip()


def main():
    files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith('.pdf')]
    if not files:
        print("No PDFs found in:", RAW_DIR)
        print("Place PDFs there and run this script to generate text files in Data/ml_texts/")
        return

    meta_rows = []
    if os.path.exists(META_FILE):
        with open(META_FILE, 'r', encoding='utf-8') as mf:
            reader = csv.DictReader(mf)
            for r in reader:
                meta_rows.append(r)

    for fn in tqdm(files, desc="Processing PDFs"):
        src = os.path.join(RAW_DIR, fn)
        out_name = os.path.splitext(fn)[0] + '.txt'
        out_path = os.path.join(OUT_DIR, out_name)

        raw = extract_text_from_pdf(src)
        cleaned = clean_text(raw)

        with open(out_path, 'w', encoding='utf-8') as of:
            of.write(cleaned)

        # Add simple metadata if not present
        if not any(r.get('filename') == fn for r in meta_rows):
            meta_rows.append({
                'filename': fn,
                'title': os.path.splitext(fn)[0],
                'authors': '',
                'year': '',
                'source': '',
                'url': '',
                'license': ''
            })

    # write metadata back
    if meta_rows:
        with open(META_FILE, 'w', encoding='utf-8', newline='') as mf:
            fieldnames = ['filename', 'title', 'authors', 'year', 'source', 'url', 'license']
            writer = csv.DictWriter(mf, fieldnames=fieldnames)
            writer.writeheader()
            for r in meta_rows:
                writer.writerow(r)

    print('Done. Text files written to', OUT_DIR)


if __name__ == '__main__':
    main()
