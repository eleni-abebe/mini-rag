from app.embeddings import embed_text
from app.retrieval import cosine_similarity, retrieve


def test_similarity_order_ranks_relevant_first():
    documents = [
        "The Eiffel Tower is located in Paris, France.",
        "Bananas are a good source of potassium.",
        "Paris is the capital city of France.",
    ]
    results = retrieve("Where is the Eiffel Tower?", documents, embed_text, top_k=3)
    top_doc, _ = results[0]
    assert "Eiffel Tower" in top_doc


def test_cosine_similarity_range_and_self_similarity():
    v = embed_text("continuous integration and deployment")
    assert cosine_similarity(v, v) > 0.99  # a vector matches itself closely
    zero_sim = cosine_similarity(v, embed_text("") * 0)
    assert -1.0 <= zero_sim <= 1.0
