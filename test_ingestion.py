from app.ingestion import read_upload, make_parent_child_chunks

def test_text_ingestion_and_parent_child_chunks():
    pages = read_upload("notes.txt", b"RAG combines retrieval with generation. " * 30)
    parents, children = make_parent_child_chunks(pages, "notes.txt", child_size=80, parent_size=200)
    assert parents
    assert children
    assert all(c["parent_id"] in {p["id"] for p in parents} for c in children)

def test_rejects_unsupported_file():
    try:
        read_upload("data.csv", b"a,b")
    except ValueError as exc:
        assert "Supported formats" in str(exc)
    else:
        assert False, "Expected ValueError"
