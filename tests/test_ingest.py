from ingest import run_ingestion                                    # :contentReference[oaicite:3]{index=3}
from rag import RagChat                                             # :contentReference[oaicite:4]{index=4}

def test_run_ingestion_returns_rag(dummy_pdf_dir):
    rag = run_ingestion(dummy_pdf_dir)
    assert isinstance(rag, RagChat)
