import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.loader import build_chunks, save_chunks

PDF_PATH = "data/raw/spotify_web_app_architecture.pdf"

docs = build_chunks(PDF_PATH)
save_chunks(docs)

print(f"✅ {len(docs)} chunks saved to data/processed/chunks.json\n")
for d in docs:
    m = d.metadata
    print(f"{m['chunk_id']:<18} p{m['page']}  {len(d.page_content):>5} chars  "
          f"{m['section']} > {m['subsection']}")
