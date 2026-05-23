"""Streamlit demo for the multimodal recipe recommender.

Run: streamlit run src/app.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision import analyze_image
from src.speech import transcribe_wav
from src.retrieval import RecipeRetriever
from src.recommender import recommend


@st.cache_resource
def _retriever():
    return RecipeRetriever.load_default()


st.set_page_config(page_title="What's in my fridge?", page_icon="🥗", layout="wide")
st.title("🥗 What's in my fridge?")
st.caption("SIT788 11.2HD — Multimodal Recipe Recommender (Azure AI Vision + Speech + OpenAI)")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Snap your fridge")
    image_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
with col2:
    st.subheader("2. Tell me what you want")
    voice = st.audio_input("Record (or skip and type below)")
    typed = st.text_input("...or type preferences", placeholder="vegetarian, under 30 minutes")

run = st.button("Recommend recipes", type="primary", use_container_width=True)

if run:
    if image_file is None:
        st.error("Upload a fridge photo first.")
        st.stop()

    with st.status("Analysing image with Azure AI Vision...", expanded=True) as status:
        vis = analyze_image(image_file.read())
        st.write(f"**Caption:** {vis.caption}")
        st.write(f"**Detected ingredients:** {', '.join(vis.ingredients) or '(none above threshold)'}")
        status.update(label="Vision done", state="complete")

    if voice is not None:
        with st.status("Transcribing voice with Azure Speech...", expanded=True) as status:
            spoken = transcribe_wav(voice.getvalue())
            st.write(f"**You said:** {spoken or '(no speech detected)'}")
            status.update(label="Speech done", state="complete")
    else:
        spoken = ""

    prefs = " | ".join(p for p in (spoken, typed) if p)

    with st.status("Retrieving candidates from FAISS index...", expanded=True) as status:
        query = f"recipe using {', '.join(vis.ingredients)}. {prefs}"
        candidates = _retriever().search(query, k=10)
        st.write(f"Pulled {len(candidates)} candidate recipes.")
        status.update(label="Retrieval done", state="complete")

    with st.status("Asking GPT-5-mini to pick the best 3...", expanded=True) as status:
        picks = recommend(vis.ingredients, prefs, candidates)
        status.update(label="Recommendations ready", state="complete")

    st.divider()
    st.subheader("🍽️ Your three recipes")
    by_id = {c["id"]: c for c in candidates}
    for rec in picks:
        full = by_id.get(rec.get("recipe_id"))
        with st.container(border=True):
            st.markdown(f"### {rec.get('name')}")
            st.markdown(f"_{rec.get('why')}_")
            if full:
                st.markdown(f"**Time:** {full['minutes']} min  |  **Match score:** {full['score']:.2f}")
                with st.expander("Ingredients"):
                    st.write(", ".join(full["ingredients"]))
                with st.expander("Steps"):
                    for i, step in enumerate(full["steps"], 1):
                        st.write(f"{i}. {step}")
