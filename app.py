# app.py
import os, uuid, threading
from pathlib import Path

from flask import (
    Flask, render_template, request,
    redirect, url_for, jsonify, session
)
from werkzeug.utils import secure_filename

from ingest import run_ingestion                   # wrapper around your pipeline

# ──────────────────────────── config ────────────────────────────
ALLOWED_EXT = {"pdf"}
UPLOAD_DIR  = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev")    # session signing key

# In-memory job registry: job_id -> dict(status, rag, error)
jobs: dict[str, dict] = {}

def _allowed(name: str) -> bool:
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# ──────────────────────────── routes ────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("files[]")
        if not files:
            return render_template("index.html",
                                   error="Please choose at least one PDF.")

        job_id   = str(uuid.uuid4())
        save_dir = UPLOAD_DIR / job_id
        save_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            if _allowed(f.filename):
                f.save(save_dir / secure_filename(f.filename))

        # register + spawn worker
        jobs[job_id] = {"status": "pending", "rag": None, "error": None}
        threading.Thread(target=_run_job,
                         args=(job_id, save_dir),
                         daemon=True).start()

        session["job_id"] = job_id
        return redirect(url_for("progress", job_id=job_id))

    return render_template("index.html")

# ───────────────── background worker ────────────────────────────
def _run_job(job_id: str, folder: Path):
    """
    Heavy ingestion happens here.  IMPORTANT: we insert the RagChat
    object first, then flip status to 'done'—avoids race condition.
    """
    try:
        rag = run_ingestion(folder)           # ← your pipeline
        jobs[job_id]["rag"] = rag             # ① store object
        jobs[job_id]["status"] = "done"       # ② NOW mark done
    except Exception as exc:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(exc)

# ───────────────── progress / status ────────────────────────────
@app.route("/progress/<job_id>")
def progress(job_id):
    return render_template("progress.html", job_id=job_id)

@app.route("/status/<job_id>")
def status(job_id):
    """
    Only return JSON-serialisable fields so jsonify never crashes.
    """
    j = jobs.get(job_id, {})
    return jsonify({
        "status": j.get("status"),      # pending | done | error | None
        "error" : j.get("error")        # optional message
    })

# ───────────────── chat UI & API ────────────────────────────────
@app.route("/chat/<job_id>")
def chat(job_id):
    if jobs.get(job_id, {}).get("status") != "done":
        return redirect(url_for("progress", job_id=job_id))
    return render_template("chat.html", job_id=job_id)

@app.route("/api/chat/<job_id>", methods=["POST"])
def chat_api(job_id):
    data      = request.get_json(force=True)
    question  = data.get("message", "")

    # ----- robust check that the job exists and ingestion finished -----
    job = jobs.get(job_id)           # None if unknown job_id
    rag = job.get("rag") if job else None
    if rag is None:                  # covers: job unknown, still pending, or errored
        return jsonify({
            "ok": False,
            "answer": "Ingestion not finished."
        })

    # -------------------------------------------------------------------
    answer_html = rag.answer(question)
    return jsonify({"ok": True, "answer": answer_html})

# ───────────────── local dev entry-point ────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
