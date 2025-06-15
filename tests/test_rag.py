from rag import clean_query, md_table_to_html

def test_clean_query():
    # If every token is a stop-word the function should fall back
    # to the original string (current rag.py behaviour).
    assert clean_query("Figure of the diagram") == "Figure of the diagram"

def test_md_table_roundtrip():
    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    html = md_table_to_html(md)
    assert "<table" in html and "<td" in html
