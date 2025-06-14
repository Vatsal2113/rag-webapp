from rag import clean_query, md_table_to_html                       # :contentReference[oaicite:2]{index=2}

def test_clean_query():
    assert clean_query("Figure of the diagram") == "diagram"

def test_md_table_roundtrip():
    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    html = md_table_to_html(md)
    assert "<table" in html and "<td" in html
