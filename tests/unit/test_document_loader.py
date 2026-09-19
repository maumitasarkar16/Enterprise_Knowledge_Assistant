import pytest
from src.document_loader import load_documents

def test_load_txt_and_md(tmp_path):
    (tmp_path/"a.txt").write_text("hello txt", encoding="utf-8")
    (tmp_path/"b.md").write_text("# hello md", encoding="utf-8")
    docs = load_documents(str(tmp_path))
    assert {d.metadata["source"] for d in docs} == {"a.txt","b.md"}

def test_empty_and_unsupported_files_are_ignored(tmp_path):
    (tmp_path/"empty.txt").write_text("", encoding="utf-8")
    (tmp_path/"image.jpg").write_bytes(b"fake")
    (tmp_path/"valid.txt").write_text("valid content", encoding="utf-8")
    docs = load_documents(str(tmp_path))
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "valid.txt"

def test_missing_directory_raises():
    with pytest.raises(FileNotFoundError):
        load_documents("/definitely/not/a/real/folder")

def test_only_empty_documents_raises(tmp_path):
    (tmp_path/"empty.txt").write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_documents(str(tmp_path))
