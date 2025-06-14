from app import _allowed_file, jobs

def test_allowed_file():
    assert _allowed_file("report.pdf")
    assert not _allowed_file("virus.exe")

def test_status_endpoint(client):
    jobs["abc"] = {"status": "pending", "error": None}
    rv = client.get("/status/abc")
    assert rv.status_code == 200
    assert rv.get_json() == {"status": "pending", "error": None}
