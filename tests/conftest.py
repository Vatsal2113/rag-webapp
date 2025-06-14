import pytest, pathlib
from unittest import mock
from app import app

@pytest.fixture
def client():
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c

# automatically stub out external services
@pytest.fixture(autouse=True)
def no_external(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setattr("ingest.google.generativeai.GenerativeModel", mock.Mock)
    monkeypatch.setattr("ingest.HuggingFaceEmbeddings", mock.Mock)

# place a tiny dummy PDF in a temp dir for ingest tests
@pytest.fixture
def dummy_pdf_dir(tmp_path: pathlib.Path):
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(
        b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 2\n0000000000 65535 f\n"
        b"0000000010 00000 n\ntrailer\n<<>>\nstartxref\n45\n%%EOF"
    )
    return tmp_path
