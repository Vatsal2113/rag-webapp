# ─────────────────────────── tests/conftest.py ───────────────────────────
import pathlib, pytest, sys, types
from unittest import mock
from app import app

# ------------------------------------------------------------------
# Flask client
# ------------------------------------------------------------------
@pytest.fixture
def client():
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c

# ------------------------------------------------------------------
# Stub external services automatically for every test
# ------------------------------------------------------------------
@pytest.fixture(autouse=True)
def no_external(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    # --- Stub google.generativeai.GenerativeModel -----------------
    fake_google = types.ModuleType("google")
    fake_genai  = types.ModuleType("generativeai")
    fake_genai.GenerativeModel = mock.Mock
    fake_google.generativeai = fake_genai
    sys.modules["google"] = fake_google
    sys.modules["google.generativeai"] = fake_genai

    # --- Stub embeddings model used in ingest.py ------------------
    # ⬇️ the only line that changed — now uses raising=False
    monkeypatch.setattr("ingest.HuggingFaceEmbeddings",
                        mock.Mock,
                        raising=False)

# ------------------------------------------------------------------
# Tiny dummy-PDF fixture
# ------------------------------------------------------------------
@pytest.fixture
def dummy_pdf_dir(tmp_path: pathlib.Path):
    (tmp_path / "dummy.pdf").write_bytes(
        b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 2\n0000000000 65535 f\n"
        b"0000000010 00000 n\ntrailer\n<<>>\nstartxref\n45\n%%EOF"
    )
    return tmp_path
