# ingest.py
"""
Ingest PDFs → extract text / tables / images → embed into Chroma vector
store → return a RagChat object for the Flask app.

All heavy lifting happens in run_ingestion(pdf_folder).
"""

# ── 0. Std-lib tweaks & warnings ---------------------------------
import os, re, difflib, sqlite3, shutil, warnings
from pathlib import Path
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"      # kill fork spam
warnings.filterwarnings("ignore",
    message="Token indices sequence length is longer", category=UserWarning)

# ── 1. Third-party imports ---------------------------------------
from PIL import Image
import fitz, pytesseract, tiktoken, google.generativeai as genai

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from transformers import AutoTokenizer
from langchain_community.vectorstores import Chroma         # new import path (>=0.4)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from rag import RagChat

# ── 2. Helper functions ------------------------------------------
enc = tiktoken.get_encoding("cl100k_base")

def n_tok(txt: str) -> int:
    return len(enc.encode(txt))

# roman numerals → int
ROMAN_MAP = dict(zip("ivxlcdm", (1,5,10,50,100,500,1000)))
def roman_to_int(s: str) -> int:
    total, prev = 0, 0
    for ch in reversed(s.lower()):
        val = ROMAN_MAP.get(ch, 0)
        total += -val if val < prev else val
        prev = val
    return total

# normalise "Figure IIIb" → "fig3b"
LABEL_RE = re.compile(r'^(fig(?:ure)?|table)\s*([ivxlcdm]+|\d+)([a-z])?', re.I)
def label_key(txt: str) -> str | None:
    if not txt: return None
    m = LABEL_RE.match(txt.lower().replace(" ", ""))
    if not m: return None
    head, raw, suff = m.groups()
    num = raw if raw.isdigit() else str(roman_to_int(raw))
    key = ("fig" if head.startswith("fig") else "table") + num
    return key + suff if suff else key

def clean(text: str) -> str:
    text = text.replace("\u00ad", "")                 # soft-hyphen
    text = re.sub(r'(?<=\w)-\n(?=\w)', "", text)      # de-hyphenate
    return re.sub(r'\s*\n\s*', " ", text).strip()

def ocr_image(im: Image.Image) -> str:
    return clean(pytesseract.image_to_string(im, lang="eng"))

def pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    mode = "RGBA" if pix.alpha else "RGB"
    return Image.frombytes(mode, (pix.width, pix.height), pix.samples)

# ── 3. Main entry function ---------------------------------------
def run_ingestion(pdf_folder: str | Path) -> RagChat:
    pdf_folder = Path(pdf_folder)
    GEM_KEY    = os.getenv("GEMINI_API_KEY")
    if not GEM_KEY:
        raise RuntimeError("GEMINI_API_KEY env-var missing")

    # — paths in this job —
    OUT_IMG_DIR = pdf_folder / "_pics"
    DB_PATH     = pdf_folder / "chunks.db"
    STORE_DIR   = pdf_folder / "chroma_store"

    # clean leftovers from previous job
    for p in (DB_PATH,):
        if p.exists(): p.unlink()
    for d in (STORE_DIR, OUT_IMG_DIR):
        shutil.rmtree(d, ignore_errors=True)
    OUT_IMG_DIR.mkdir(exist_ok=True)

    # — Gemini —
    genai.configure(api_key=GEM_KEY)
    GEM_MM = genai.GenerativeModel("gemini-1.5-flash-latest")

    # — SQLite registry —
    conn = sqlite3.connect(DB_PATH)
    conn.create_function("REGEXP", 2,
        lambda expr, itm: 1 if itm and re.search(expr, itm) else 0)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE chunks(
      chunk_id INTEGER PRIMARY KEY,
      source   TEXT, page INTEGER, type TEXT,
      content  TEXT, caption TEXT, img_path TEXT,
      parent_chunk_id INTEGER, label_key TEXT)
    """)
    conn.commit()

    # — Embeddings & tokenizer —
    EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer  = AutoTokenizer.from_pretrained("bert-base-uncased")
    embedder   = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    # — Convert PDFs with Docling —
    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=PdfPipelineOptions(
                images_scale=2.0,
                generate_picture_images=True))
    })

    docling_docs = {}
    for pdf in pdf_folder.glob("*.pdf"):
        print("→ parsing", pdf.name, flush=True)
        docling_docs[pdf.name] = converter.convert(str(pdf)).document

    # — Chunking / extraction —
    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=1024, stride=200)
    cid = 0
    FIG_TBL = re.compile(r'(fig(?:ure)?\.?\s*\d+[a-z]?|table\s+[ivxlcdm\d]+[a-z]?)', re.I)

    for src, doc in docling_docs.items():
        stem    = Path(src).stem.lower()
        img_dir = OUT_IMG_DIR / stem
        img_dir.mkdir(exist_ok=True)

        # 1. text & equations
        for ch in chunker.chunk(doc):
            page = ch.meta.doc_items[0].prov[0].page_no
            txt  = clean(ch.text)
            typ  = ("equation" if len(txt) < 300 and
                    re.search(r"(\\frac|\\sum|\\int|=|[∑∫√±×÷])", txt)
                    else "text")
            cid += 1
            cur.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
                (cid, stem, page, typ, txt, None, None, None, None))

        # 2. OCR fallback for empty pages
        pdfdoc = fitz.open(str(pdf_folder / src))
        for p in range(1, len(doc.pages)+1):
            if not cur.execute("SELECT 1 FROM chunks WHERE source=? AND page=? AND type='text'",
                               (stem, p)).fetchone():
                pil = pixmap_to_pil(pdfdoc[p-1].get_pixmap(dpi=300))
                page_txt = clean(pytesseract.image_to_string(pil, lang="eng"))
                cid += 1
                cur.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
                    (cid, stem, p, "page", page_txt, None, None, None, None))
        pdfdoc.close()

        # 3. tables
        tbl_no = 0
        for tbl in doc.tables:
            if not tbl.prov: continue
            tbl_no += 1
            page = tbl.prov[0].page_no
            md   = tbl.export_to_markdown(doc)
            if not FIG_TBL.search(md):
                md = f"table{tbl_no}:\n{md}"
            summary = GEM_MM.generate_content(
                ["Write a one-sentence caption for this table:", md]).text.strip()
            cap = f"table{tbl_no}: {summary}"
            cid += 1
            cur.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
                (cid, stem, page, "table", md, cap, None, None, label_key(cap)))

        # 4. images
        pdfdoc = fitz.open(str(pdf_folder / src))
        fig_no = 0
        for i, pic in enumerate(doc.pictures, 1):
            if not pic.prov: continue
            fig_no += 1
            page = pic.prov[0].page_no
            pil  = pic.get_image(doc)
            fp   = img_dir / f"fig{fig_no}_p{page}_{i}.png"
            pil.save(fp, "PNG")
            cid += 1
            cur.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
                (cid, stem, page, "image",
                 ocr_image(pil), None, str(fp), None,
                 label_key(f"fig{fig_no}:")))
        pdfdoc.close()
        conn.commit()

    # 5. summarise all images
    cur.execute("SELECT chunk_id, img_path FROM chunks WHERE type='image'")
    for cid_, img_path in cur.fetchall():
        cap = GEM_MM.generate_content(
            ["Write a concise one-sentence summary of this figure:",
             Image.open(img_path)]).text.strip()
        cur.execute("UPDATE chunks SET caption=? WHERE chunk_id=?", (cap, cid_))
    conn.commit()
    print("✅ All image captions populated.", flush=True)

    # ── Build Chroma vector store (auto-persists) —
    docs = []
    for cid_, src, pg, typ, txt, cap in cur.execute(
        "SELECT chunk_id, source, page, type, content, caption FROM chunks"):
        full = ((cap or "") + "\n" + txt) if typ in ("image", "table") else txt
        docs.append(Document(full, metadata={
            "chunk_id": cid_, "source": src, "page": pg, "type": typ}))

    vectordb = Chroma(
        collection_name="multimodal_rag",
        persist_directory=str(STORE_DIR),
        embedding_function=embedder)
    vectordb.add_documents(docs)  # auto-saved; no .persist()

    # helper for Flask
    def fetch(cid_: int):
        return cur.execute(
            "SELECT source, page, type, content, caption, img_path "
            "FROM chunks WHERE chunk_id=?", (cid_,)).fetchone()

    # lightweight answer fn
    def answer(q: str) -> str:
        hits = vectordb.similarity_search(q, k=6, filter={"type": "text"})
        ctx  = "\n\n".join(h.page_content for h in hits)
        prompt = [f"Use the context below to answer the question."
                  f"\n\nContext:\n{ctx}\n\nQ: {q}\nA:"]
        return GEM_MM.generate_content(prompt).text.strip()

    return RagChat(vectordb, fetch, answer)
