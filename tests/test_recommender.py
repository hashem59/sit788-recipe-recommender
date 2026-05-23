from unittest.mock import MagicMock
from src.recommender import build_prompt, recommend


def test_build_prompt_includes_ingredients_and_prefs():
    msgs = build_prompt(
        ingredients=["tomato", "onion", "garlic"],
        preferences="vegetarian, under 30 minutes",
        candidates=[
            {"id": 1, "name": "Pasta Marinara", "ingredients": ["pasta", "tomato"], "minutes": 25, "score": 0.81},
            {"id": 2, "name": "Beef Stew", "ingredients": ["beef"], "minutes": 120, "score": 0.41},
        ],
    )
    blob = "\n".join(m["content"] for m in msgs).lower()
    assert "tomato" in blob and "onion" in blob and "garlic" in blob
    assert "vegetarian" in blob
    assert "pasta marinara" in blob
    assert "beef stew" in blob


def test_recommend_returns_three_recipes(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=(
            '{"recommendations": ['
            '{"recipe_id": 1, "name": "A", "why": "x"},'
            '{"recipe_id": 2, "name": "B", "why": "y"},'
            '{"recipe_id": 3, "name": "C", "why": "z"}]}'
        )))]
    )
    monkeypatch.setattr("src.recommender.openai_client", lambda: fake_client)
    out = recommend(
        ingredients=["egg"],
        preferences="quick",
        candidates=[{"id": i, "name": f"R{i}", "ingredients": [], "minutes": 10, "score": 0.5} for i in range(1, 11)],
    )
    assert len(out) == 3
    assert out[0]["name"] == "A"
    assert "why" in out[0]
