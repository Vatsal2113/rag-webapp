# rag.py
"""
Thin wrapper around the vector store so Flask only calls .answer().
"""

from __future__ import annotations
from pathlib import Path
from base64 import b64encode
import re, html
from typing import Callable, Any

# ─────────────────── helpers ───────────────────────────────────────────

STOP_WORDS = {
    "figure", "fig", "image", "diagram", "picture",
    "table", "chart", "plot",
    "of", "for", "showing", "displaying", "the", "a", "an"
}

def clean_query(q: str) -> str:
    """
    Strip filler words so similarity search focuses on informative tokens.
    """
    tokens = re.split(r'\W+', q.lower())
    keep   = [t for t in tokens if t and t not in STOP_WORDS]
    return " ".join(keep) or q   # fall back if we stripped everything

def b64_img(img_path: str | Path) -> str:
    """Return base-64 data URI for an image file."""
    return b64encode(Path(img_path).read_bytes()).decode()

def md_table_to_html(md: str) -> str:
    """
    Convert a GitHub-style pipe table to an HTML <table> with borders.
    If parsing fails, fall back to <pre>.
    """
    lines = [l.strip() for l in md.splitlines() if "|" in l]
    if not lines:                       # no pipes → preformatted
        return f"<pre>{html.escape(md)}</pre>"

    def split(row):                     # trim leading/trailing '|'
        return [c.strip() for c in row.strip("|").split("|")]

    head, *body = lines
    html_rows = ["<thead><tr>" + "".join(
        f"<th style='border:1px solid #999;padding:4px;'>"
        f"{html.escape(c)}</th>" for c in split(head)) + "</tr></thead>"]

    for b in body:
        if re.fullmatch(r"\s*[:\-| ]+\s*", b):  # skip separator row
            continue
        html_rows.append("<tr>" + "".join(
            f"<td style='border:1px solid #999;padding:4px;'>"
            f"{html.escape(c)}</td>" for c in split(b)) + "</tr>")

    return ("<table style='border-collapse:collapse;width:100%;"
            "margin:.6em 0;border:1px solid #999;'>" +
            "".join(html_rows) + "</table>")

# ─────────────────── RagChat class ─────────────────────────────────────

class RagChat:
    """
    A minimal surface area for the Flask app.
    """

    def __init__(
        self,
        vectordb,
        fetch_fn: Callable[[int], tuple[Any, ...]],
        llm_answer_fn: Callable[[str], str]
    ):
        """
        vectordb ........ LangChain vector store
        fetch_fn ........ fetch(cid) -> (src, page, type, content, caption, img_path)
        llm_answer_fn ... fallback textual answer generator
        """
        self.db      = vectordb
        self._fetch  = fetch_fn
        self._llm    = llm_answer_fn

    # ────────── public API ────────────────────────────────────────────
    def answer(self, question: str) -> str:
        q_lower = question.lower()

        # --- explicit "figure ..." or label like "Figure 2" -------------
        label_match = re.search(
            r'\b(fig(?:ure)?|table)\s*([ivxlcdm\d]+[a-z]?)\b', q_lower)
        if label_match:
            kind  = "image" if label_match.group(1).startswith("fig") else "table"
            return self._inject_by_label(label_match.group(0), kind)

        # --- explicit requests for a figure/table description ----------
        if any(w in q_lower for w in ("figure", "image", "diagram", "fig ")):
            return self._best_match_block(question, "image")
        if "table" in q_lower:
            return self._best_match_block(question, "table")

        # --- fallback: normal LLM answer with inline injections --------
        html_answer = self._llm(question).replace("\n", "<br>")
        html_answer = re.sub(
            r'\b(fig(?:ure)?|table)\s*([ivxlcdm\d]+[a-z]?)\b',
            lambda m: self._inject_by_label(m.group(0)),
            html_answer, flags=re.I
        )
        return html_answer

    # ────────── internal helpers ───────────────────────────────────────
    def _best_match_block(self, query: str, kind: str) -> str:
        """
        Return ONE best-matching figure or table rendered nicely.
        """
        hit = self._top_hit(query, kind)
        if not hit:
            return f"<em>No {kind} matches that description.</em>"

        cid = hit.metadata["chunk_id"]
        _src, _page, _t, content, caption, img_path = self._fetch(cid)

        if kind == "image":
            img_data = b64_img(img_path)
            return (f"<figure>"
                    f"<img src='data:image/png;base64,{img_data}' "
                    f"alt='{caption}' style='max-width:100%;height:auto;'>"
                    f"<figcaption>{caption}</figcaption></figure>")
        else:
            table_html = md_table_to_html(content)
            return (f"<details open>"
                    f"<summary>{caption}</summary>{table_html}</details>")

    def _inject_by_label(self, label: str, kind: str | None = None) -> str:
        """
        Replace 'Figure 3b', 'Table II' in an answer with the actual object.
        """
        kind = kind or ("image" if label.lower().startswith("fig") else "table")
        key  = re.sub(r'[^ivxlcdm0-9a-z]', '', label.split()[-1], flags=re.I)

        hit = self._top_hit(key, kind)
        if not hit:
            return label                       # leave as plain text

        cid = hit.metadata["chunk_id"]
        _src, _page, _t, content, caption, img_path = self._fetch(cid)

        if kind == "image":
            return (f"<figure>"
                    f"<img src='data:image/png;base64,{b64_img(img_path)}' "
                    f"alt='{caption}' style='max-width:100%;height:auto;'>"
                    f"<figcaption>{caption}</figcaption></figure>")
        else:
            table_html = md_table_to_html(content)
            return (f"<details open>"
                    f"<summary>{caption}</summary>{table_html}</details>")

    def _top_hit(self, query: str, kind: str):
        """
        Clean the query → similarity search → return top document or None.
        """
        cleaned = clean_query(query)
        hits = self.db.similarity_search(cleaned, k=1, filter={"type": kind})
        return hits[0] if hits else None

    # so it prints nicely in logs
    def __repr__(self):
        return f"<RagChat {hex(id(self))}>"
