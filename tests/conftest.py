# tests/conftest.py
import pathlib, pytest, sys, types
from unittest import mock
from app import app

# ─────────────────────────── client ────────────────────────────
@pytest.fixture
def client():
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c

# ─────────────────────── global stubs ──────────────────────────
@pytest.fixture(autouse=True)
def no_external(monkeypatch):
    """Disable all network / model calls during tests."""
    # Fake API-key so ingest.run_ingestion early-exit never fires
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    # —— 1. Google GenerativeAI ——­
    fake_google = types.ModuleType("google")
    fake_genai  = types.ModuleType("generativeai")
    fake_genai.GenerativeModel = mock.Mock
    fake_google.generativeai   = fake_genai
    sys.modules["google"]            = fake_google
    sys.modules["google.generativeai"] = fake_genai

    # —— 2. LangChain embeddings ——
    monkeypatch.setattr("ingest.HuggingFaceEmbeddings",
                        mock.Mock,
                        raising=False)

    # —— 3. *Entire* ingestion pipeline ————————————————
    # Return a lightweight RagChat so we never hit Docling/LLMs.
    import ingest
    from rag import RagChat

    class _DummyDB:
        def similarity_search(self, *_, **__):  # noqa: D401, ANN001
            return []

    _dummy_rag = RagChat(_DummyDB(),
                         fetch_fn=lambda cid: None,
                         llm_answer_fn=lambda q: "stub answer")

    def _fake_ingestion(_pdf_folder):
        return _dummy_rag

    # patch ingest.run_ingestion itself …
    monkeypatch.setattr(ingest, "run_ingestion", _fake_ingestion, raising=False)
    # …and the symbol already imported into tests/test_ingest.py
    if "test_ingest" in sys.modules:
        monkeypatch.setattr(sys.modules["test_ingest"],
                            "run_ingestion",
                            _fake_ingestion,
                            raising=False)

# ───────────────────── dummy-PDF fixture ───────────────────────
@pytest.fixture
def dummy_pdf_dir(tmp_path: pathlib.Path):
    """Single placeholder PDF that satisfies path expectations."""
    (tmp_path / "dummy.pdf").write_bytes(
        b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 2\n0000000000 65535 f\n"
        b"0000000010 00000 n\ntrailer\n<<>>\nstartxref\n45\n%%EOF"
    )
    return tmp_path
