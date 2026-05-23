# Panopto Demo — 4 minute script

Record at 1080p / 30fps. Use QuickTime (`Cmd+Shift+5` → record selected portion).

| Time | Scene | Talking points |
|---|---|---|
| 0:00–0:30 | Title slide + intro | "Hashe, s225048421, SIT788 Task 11.2HD. This is a multimodal recipe recommender that turns a phone photo of your fridge plus a voice query into three ranked recipe suggestions, powered by four Azure AI services." |
| 0:30–1:00 | `docs/architecture.png` on screen | Narrate the five boxes left-to-right: Streamlit captures image + voice → Vision tags the fridge → Speech transcribes the query → embedder + FAISS retrieve 10 candidates → GPT-5-mini ranks the top 3 with reasoning. |
| 1:00–1:30 | Terminal showing `az resource list -g deakinuni -o table` | Point to `sit788cv2srbwq` (Vision), `sit788-s225048421-speech` (Speech), `sit788-s225048421-openai` (OpenAI). "All four services run under my student lab subscription — Cognitive Services Contributor role, existing `deakinuni` resource group." |
| 1:30–2:30 | Live Streamlit demo | Upload the fridge photo. Record voice: *"Something vegetarian and under thirty minutes."* Click Recommend. Walk through each status card as it expands — Vision caption + ingredients, Speech transcript, retrieval pulled 10 candidates, GPT-5-mini picked these three. |
| 2:30–3:15 | Show the three recipe cards | For each one, expand Ingredients and Steps. Highlight that the *why* sentence references both detected ingredients and the spoken preference. "Notice the model excluded the carbonara because it's not vegetarian — the soft preference filtering is happening at the LLM stage." |
| 3:15–3:45 | Code in VS Code | Open `src/recommender.py` — show the system prompt. Open `src/retrieval.py` — show the FAISS search call. "GPT-5-mini deprecated `temperature`; uses `max_completion_tokens`; both quirks documented inline." |
| 3:45–4:00 | Outro | "Limitations: corpus is US-centric, tags are coarse. Future work: domain-tuned Custom Vision model and Azure AI Search at production scale. Thank you." |

## Panopto upload checklist

1. Upload to https://deakin.au.panopto.com (use Deakin SSO).
2. Set visibility to **"Anyone at your organization with the link"** (NOT "Specific people").
3. **Test the link in a private/incognito tab signed out of Deakin** to confirm it plays.
4. Copy the share link into `docs/report.md` Section 0 (Panopto Demo).
5. Re-export `docs/report.pdf` after editing.
