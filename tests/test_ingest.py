# ──────────────────────────── tests/test_ingest.py ───────────────────────
from ingest import run_ingestion
from rag import RagChat

def test_run_ingestion_returns_rag(dummy_pdf_dir):
    rag = run_ingestion(dummy_pdf_dir)
    assert isinstance(rag, RagChat)
