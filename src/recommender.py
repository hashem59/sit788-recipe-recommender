"""GPT-5-mini recommender: rank top candidates and explain.

Note on gpt-5-mini params:
  - Uses `max_completion_tokens`, not `max_tokens`.
  - Only supports default temperature (1.0); cannot be overridden.
  - Reasoning tokens are charged but not returned; budget accordingly.
"""
from __future__ import annotations
import json
from .config import Settings, openai_client

SYSTEM = (
    "You are a culinary assistant. Given a list of ingredients the user has on hand, "
    "their stated preferences, and 10 candidate recipes retrieved from a corpus, "
    "select the THREE best recipes for them. For each pick, give a one-sentence "
    "reason that references both the ingredients and the preferences. "
    "Respond ONLY as JSON of the form:\n"
    '{"recommendations":[{"recipe_id":<int>,"name":"<str>","why":"<str>"}, ...]}'
)


def _format_candidate(c: dict) -> str:
    ings = ", ".join(c["ingredients"][:10])
    mins = c.get("minutes") or "?"
    return f"- id={c['id']} | {c['name']} | {mins} min | ingredients: {ings} | score={c['score']:.2f}"


def build_prompt(ingredients: list[str], preferences: str, candidates: list[dict]) -> list[dict]:
    user = (
        f"INGREDIENTS I HAVE: {', '.join(ingredients) or '(none detected)'}\n"
        f"PREFERENCES: {preferences or '(none)'}\n\n"
        f"CANDIDATE RECIPES:\n" + "\n".join(_format_candidate(c) for c in candidates)
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def recommend(ingredients: list[str], preferences: str, candidates: list[dict]) -> list[dict]:
    msgs = build_prompt(ingredients, preferences, candidates)
    resp = openai_client().chat.completions.create(
        model=Settings.chat_deployment,
        messages=msgs,
        response_format={"type": "json_object"},
        max_completion_tokens=2000,
    )
    data = json.loads(resp.choices[0].message.content)
    recs = data.get("recommendations", [])[:3]
    return recs
