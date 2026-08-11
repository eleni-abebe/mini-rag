from app.chunking import chunk_text


def test_chunk_text_basic_splits():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) == 3
    assert chunks[0].split()[0] == "word0"


def test_chunk_text_overlap():
    text = " ".join(f"word{i}" for i in range(20))
    chunks = chunk_text(text, chunk_size=10, overlap=5)
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    # last 5 words of chunk 0 should equal first 5 words of chunk 1
    assert first_words[-5:] == second_words[:5]


def test_chunk_text_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []
