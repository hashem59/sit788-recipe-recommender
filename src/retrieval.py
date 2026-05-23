"""Recipe retrieval over a pre-built FAISS index."""
from __future__ import annotations
import ast
from dataclasses import dataclass
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from .config import Settings, openai_client

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "artifacts" / "index.faiss"
META_PATH = ROOT / "artifacts" / "recipes_meta.parquet"


def _parse_list(val) -> list[str]:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return list(ast.literal_eval(val))
        except (ValueError, SyntaxError):
            return []
    return []


@dataclass
class RecipeRetriever:
    index: faiss.Index
    meta: pd.DataFrame

    @classmethod
    def load_default(cls) -> "RecipeRetriever":
        return cls(
            index=faiss.read_index(str(INDEX_PATH)),
            meta=pd.read_parquet(META_PATH),
        )

    def _embed(self, text: str) -> np.ndarray:
        resp = openai_client().embeddings.create(
            model=Settings.embed_deployment, input=[text]
        )
        v = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(v)
        return v

    def search(self, query: str, k: int = 10) -> list[dict]:
        qv = self._embed(query)
        scores, idxs = self.index.search(qv, k)
        out = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            row = self.meta.iloc[int(i)]
            out.append({
                "id": int(row["id"]),
                "name": row["name"],
                "ingredients": _parse_list(row["ingredients_list"]),
                "steps": _parse_list(row["steps_list"]),
                "minutes": int(row["minutes"]) if pd.notna(row["minutes"]) else None,
                "score": float(score),
            })
        return out
