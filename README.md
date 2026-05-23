# What's in My Fridge? — Multimodal Recipe Recommender

SIT788 Task 11.2HD capstone — Hashe (s225048421).

A Streamlit app that takes a photo of your fridge plus a voice/text query and returns three ranked recipes, powered by four Azure AI services and a local FAISS retrieval index.

See [`docs/report.md`](docs/report.md) for the full project report and [`docs/architecture.md`](docs/architecture.md) for the system diagram.

## Architecture (one-liner)

```
photo  → Azure AI Vision 4.0 → ingredients ─┐
voice  → Azure AI Speech     → preferences ─┴─► query → text-embedding-3-small → FAISS top-10 → GPT-5-mini rerank → 3 recipes
```

## Quickstart

```bash
# 1. Python venv (3.11 required for faiss-cpu wheels)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Azure resources (one-time)
chmod +x infra/provision.sh
./infra/provision.sh
# Copy the printed key=value lines into .env (cp .env.example .env first)

# 3. Recipe corpus (one-time, requires Kaggle API token at ~/.kaggle/kaggle.json)
python data/download_dataset.py
python data/build_index.py

# 4. Run the app
streamlit run src/app.py
```

Open the browser tab Streamlit prints (default `http://localhost:8501`), upload a fridge photo, optionally record voice or type a preference, and click **Recommend recipes**.

## Tests

```bash
pytest tests/ -v                          # unit tests (no Azure calls)
TEST_AZURE_LIVE=1 pytest tests/ -v        # also runs live Azure smoke tests
```

## File map

| Path | Purpose |
|---|---|
| `src/config.py` | Loads `.env`, cached Azure clients |
| `src/vision.py` | Photo → ingredient list (Vision 4.0) |
| `src/speech.py` | WAV → transcript (Speech) |
| `src/retrieval.py` | Query → top-k recipes (FAISS + embeddings) |
| `src/recommender.py` | Candidates → 3 ranked picks (GPT-5-mini) |
| `src/app.py` | Streamlit UI |
| `data/download_dataset.py` | Pull + sample Food.com recipes |
| `data/build_index.py` | Embed sample, build FAISS index |
| `infra/provision.sh` | Azure CLI provisioning |
| `docs/report.md` | Project report (submitted as PDF) |
| `docs/architecture.md` | System diagram + service justification |
| `docs/video_script.md` | Panopto recording script |
