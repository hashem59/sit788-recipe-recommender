"""Embed the sampled recipes with Azure OpenAI and save a FAISS index.

Run: python data/build_index.py
Cost: 5k recipes * ~80 tokens each = ~400k tokens
      text-embedding-3-small ~ $0.02 per 1M tokens -> < $0.01
"""
from __future__ import annotations
import sys
from pathlib import Path
import ast
import numpy as np
import pandas as pd
import faiss
from tenacity import retry, stop_after_attempt, wait_exponential

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import Settings, openai_client  # noqa: E402

DATA_CSV = ROOT / "data" / "recipes_sample.csv"
INDEX_PATH = ROOT / "artifacts" / "index.faiss"
META_PATH = ROOT / "artifacts" / "recipes_meta.parquet"
BATCH = 100
EMBED_DIM = 1536


def doc_text(row: pd.Series) -> str:
    raw = row["ingredients_list"]
    items = ast.literal_eval(raw) if isinstance(raw, str) else raw
    ingredients = ", ".join(items)
    desc = (row["description"] or "")[:200] if isinstance(row["description"], str) else ""
    return f"{row['name']}\nIngredients: {ingredients}\n{desc}"


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=10))
def embed_batch(client, deployment: str, texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=deployment, input=texts)
    return np.array([d.embedding for d in resp.data], dtype="float32")


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    texts = [doc_text(r) for _, r in df.iterrows()]
    client = openai_client()
    deployment = Settings.embed_deployment

    all_vecs = np.empty((len(texts), EMBED_DIM), dtype="float32")
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        all_vecs[i:i + len(chunk)] = embed_batch(client, deployment, chunk)
        print(f"  embedded {i + len(chunk):,}/{len(texts):,}")

    faiss.normalize_L2(all_vecs)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(all_vecs)
    INDEX_PATH.parent.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    df.to_parquet(META_PATH, index=False)
    print(f"\nWrote {index.ntotal:,} vectors -> {INDEX_PATH}")
    print(f"Wrote metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
