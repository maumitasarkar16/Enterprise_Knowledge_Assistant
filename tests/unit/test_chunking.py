import pytest
from src.chunking import chunk_documents
from src.models import Document

def make_doc(n=30):
    return Document(
        " ".join(f"word{i}" for i in range(n)),
        {"source":"x.txt","path":"x.txt","extension":".txt"}
    )

def test_chunking_overlap_and_metadata():
    chunks = chunk_documents([make_doc()], 10, 2)
    assert len(chunks) >= 3
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[1].metadata["word_start"] == 8
    assert chunks[0].id != chunks[1].id

def test_chunk_ids_are_deterministic():
    a = chunk_documents([make_doc()], 10, 2)
    b = chunk_documents([make_doc()], 10, 2)
    assert [x.id for x in a] == [x.id for x in b]

def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_documents([make_doc()], 0, 0)

def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_documents([make_doc()], 10, 10)
