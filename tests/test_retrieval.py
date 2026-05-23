import pytest
from src.retrieval import RecipeRetriever


@pytest.fixture(scope="module")
def retriever():
    return RecipeRetriever.load_default()


def test_returns_requested_k(retriever):
    hits = retriever.search("chicken curry with rice", k=5)
    assert len(hits) == 5


def test_hits_have_required_fields(retriever):
    hits = retriever.search("chocolate cake", k=3)
    for h in hits:
        assert "name" in h and "ingredients" in h and "score" in h
        assert isinstance(h["ingredients"], list)
        assert 0.0 <= h["score"] <= 1.0


def test_ingredient_query_finds_matching_recipe(retriever):
    """A direct ingredient query should surface relevant recipes in top-10."""
    hits = retriever.search("pasta tomato garlic basil", k=10)
    names = " ".join(h["name"].lower() for h in hits)
    assert any(kw in names for kw in ["pasta", "spaghetti", "marinara", "tomato"])
