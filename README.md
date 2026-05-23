# What's in My Fridge? — Multimodal Recipe Recommender

SIT788 Task 11.2HD — Capstone submission.

See `docs/report.md` for the full report and `docs/architecture.md` for the system diagram.

## Quickstart
1. `python3.11 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in Azure keys (see `infra/provision.sh`)
4. `python data/download_dataset.py && python data/build_index.py`
5. `streamlit run src/app.py`
