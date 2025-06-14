import os, pytest, tempfile, pathlib, shutil
from unittest import mock
from app import app                                                # :contentReference[oaicite:0]{index=0}

@pytest.fixture
def client():
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c

# dummy PDF folder for ingest tests
@pytest.fixture
def dummy_pdf_dir(tmp_path: pathlib.Path):
    # put a 1-page PDF fixture there (or generate one)
    (tmp_path / "dummy.pdf").write_bytes(b"%PDF-1.4…%%EOF")
    return tmp_path

# silence Gemini and embeddings during tests
@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setattr("ingest.google.generativeai.GenerativeModel", mock.Mock)
    monkeypatch.setattr("ingest.HuggingFaceEmbeddings", mock.Mock)
