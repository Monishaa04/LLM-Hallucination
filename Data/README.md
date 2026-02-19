This folder holds authoritative machine-learning study materials used by the RAG assistant.

Guidelines
- Only add materials you have the right to store and use. For copyrighted textbooks, keep a local copy you obtained legally (publisher purchase, institutional access, or author permission).
- Prefer open-access resources when possible (arXiv papers, university lecture notes, open textbooks, MIT/Stanford OCW).
- Keep metadata (title, authors, year, source URL, license) alongside each document.

Recommended sources (open / safe to ingest):
- arXiv — https://arxiv.org (research papers; use arXiv IDs and metadata)
- Papers with Code — https://paperswithcode.com (links to code and datasets)
- Stanford CS courses (CS229, CS224N) lecture notes and slides (when publicly available)
- MIT OpenCourseWare (OCW) lecture notes and readings
- Open textbooks: "Deep Learning" (by Goodfellow — check license), "Pattern Recognition and Machine Learning" (Bishop — check license), "Machine Learning: A Probabilistic Perspective" (Murphy — check license)

How to add materials
1. Create `Data/raw_papers/` and place PDF files there (one PDF per paper/book chapter).
2. Add a small metadata file `Data/metadata.csv` with columns: `filename,title,authors,year,source,url,license`.
3. Run the ingestion script `python data/import_pdfs.py` to extract cleaned text into `Data/ml_texts/`.

Legal & ethical notes
- Do not paste large copyrighted book chapters or proprietary course materials into the repository unless you have permission.
- For copyrighted textbooks, store only metadata and a local copy outside public repos, or include small excerpts (under fair use) with attribution.

If you want, I can:
- Add an ingestion script that extracts text from PDFs and splits into chunked text files ready for embedding.
- Add utilities to fetch arXiv metadata and optionally download open-access PDFs.
