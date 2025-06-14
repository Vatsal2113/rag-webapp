import pathlib, pytest
from unittest import mock
from app import app            # ← imports app.py at repo root

# ─────────────────────────────────────────────────────────────
# Flask test-client (shared by route tests)
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def client():
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c

# ─────────────────────────────────────────────────────────────
# Auto-applied fixture that mocks external services
# ─────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def no_external(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    # Stub out expensive classes
    monkeypatch.setattr("ingest.google.generativeai.GenerativeModel", mock.Mock)
    monkeypatch.setattr("ingest.HuggingFaceEmbeddings", mock.Mock)

# ─────────────────────────────────────────────────────────────
# Tiny dummy PDF (1-object, ASCII-only) for ingest tests
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def dummy_pdf_dir(tmp_path: pathlib.Path):
    pdf = tmp_path / "dummy.pdf"
    pdf.write_bytes(
        b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 2\n0000000000 65535 f\n"
        b"0000000010 00000 n\ntrailer\n<<>>\nstartxref\n45\n%%EOF"
    )
    return tmp_path
