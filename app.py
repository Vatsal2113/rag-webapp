# app.py
import os
import uuid
import threading
import datetime
from pathlib import Path
from flask import (
    Flask, render_template, request,
    redirect, url_for, jsonify, session
)
from werkzeug.utils import secure_filename

from ingest import run_ingestion                   # your ingestion pipeline
from rag import RagChat                            # your RAG wrapper

# ─── CONFIG ──────────────────────────────────────────────────────

ALLOWED_EXT = {"pdf"}
UPLOAD_DIR  = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev")    # session signing key

jobs: dict[str, dict] = {}  # track background ingestion jobs

# ─── UTILITY FUNCTIONS ───────────────────────────────────────────

def _allowed_file(fn: str) -> bool:
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def _log_conversation(job_id: str, question: str, answer: str):
    """Append a timestamped RAG conversation entry to conversation_history.log."""
    ts = datetime.datetime.utcnow().isoformat()
    entry = f"{ts} | job_id={job_id} | Q={question!r} | A={answer!r}\n"
    p = Path("conversation_history.log")
    with p.open("a", encoding="utf-8") as f:
        f.write(entry)

# ─── ROUTES: UPLOAD & INGESTION ────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not _allowed_file(f.filename):
            return "Please upload a PDF file", 400
        fn = secure_filename(f.filename)
        dest = UPLOAD_DIR / f"{uuid.uuid4().hex}_{fn}"
        f.save(dest)

        # kick off ingestion in background
        job_id = uuid.uuid4().hex
        jobs[job_id] = {"status": "pending", "error": None, "rag": None}

        def _worker(path, jid):
            try:
                rag = run_ingestion(str(path))
                jobs[jid].update(status="done", rag=rag)
            except Exception as e:
                jobs[jid].update(status="error", error=str(e))

        threading.Thread(target=_worker, args=(dest, job_id), daemon=True).start()
        session["job_id"] = job_id
        return redirect(url_for("progress", job_id=job_id))

    return render_template("index.html")

@app.route("/progress/<job_id>")
def progress(job_id):
    return render_template("progress.html", job_id=job_id)

@app.route("/status/<job_id>")
def status(job_id):
    j = jobs.get(job_id, {})
    return jsonify({
        "status": j.get("status"),      # pending | done | error | None
        "error" : j.get("error")
    })

# ─── ROUTES: CHAT UI & API ────────────────────────────────────────

@app.route("/chat/<job_id>")
def chat(job_id):
    if jobs.get(job_id, {}).get("status") != "done":
        return redirect(url_for("progress", job_id=job_id))
    return render_template("chat.html", job_id=job_id)

@app.route("/api/chat/<job_id>", methods=["POST"])
def chat_api(job_id):
    data     = request.get_json(force=True)
    question = data.get("message", "").strip()

    job = jobs.get(job_id, {})
    rag = job.get("rag")
    if rag is None:
        return jsonify({"ok": False, "answer": "Ingestion not finished."})

    # get the answer from RAG
    answer_html = rag.answer(question)

    # **LOG THE CONVERSATION**
    _log_conversation(job_id, question, answer_html)

    return jsonify({"ok": True, "answer": answer_html})

# ─── LOCAL DEV ENTRY POINT ───────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
