from src.retrieval import tokenize

def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello, API-Key 123!") == ["hello","api","key","123"]

def test_tokenize_is_case_insensitive():
    assert tokenize("SEVERITY 1") == tokenize("severity 1")
