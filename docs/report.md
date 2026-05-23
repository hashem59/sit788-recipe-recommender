# SIT788 — Task 11.2HD: Final Project

**Student:** s225048421
**Email:** s225048421@deakin.edu.au
**Unit:** SIT788 Engineering AI Solutions
**Project Title:** "What's in My Fridge?" — Multimodal Recipe Recommender
**Submission Date:** 2026-05-31
**Panopto Demo:** _[link to be inserted after recording]_

---

## 1. Project Purpose and Target Users

Roughly one third of all food produced for human consumption is lost or wasted (FAO 2019), and a non-trivial share of household waste comes from ingredients that were forgotten in the fridge until they spoiled. This project addresses that gap by building a multimodal recommendation bot that answers the everyday question *"what should I cook with what I already have?"* — without the user needing to type out an ingredient list.

The target user is a time-pressed home cook standing in front of an open fridge. Typing on a phone in that context is awkward; speaking and snapping a photo is natural. The system accepts (a) a photo of the fridge contents, (b) a spoken or typed preference statement (e.g. *"vegetarian and under thirty minutes"*), and returns three ranked recipe suggestions, each justified against both the available ingredients and the stated preference. The multimodal entry surface is deliberately tied to the use context, not bolted on as a gimmick.

## 2. Dataset

The recommendation corpus is a 5,000-row stratified sample of the **Food.com Recipes and Interactions** dataset (Li 2019), distributed publicly on Kaggle and widely cited in recommender-system research. The full dataset contains 230,186 recipes; the sample was drawn with a fixed random seed (42) after filtering to recipes with between 4 and 15 ingredients and between 3 and 25 preparation steps. This range excludes both trivially short entries (likely incomplete records) and very long professional recipes that are unlikely matches for a home-cook query.

The 5,000-row size was chosen so that the embedding index fits in memory (~30 MB) and rebuilds in under five minutes against the Azure OpenAI `text-embedding-3-small` endpoint, costing well under one cent. The sampled CSV is committed in the repository (`data/recipes_sample.csv`) to make the build deterministic and reproducible by a marker without re-querying Kaggle.

## 3. System Architecture

The system is a single Streamlit process that orchestrates four Azure AI services across a three-stage funnel: ingredient extraction (Vision) and preference capture (Speech) feed a query builder, which embeds the joint query, retrieves the ten nearest recipes by cosine similarity from a FAISS index, and finally hands the candidates to GPT-5-mini for re-ranking and natural-language reasoning.

![Architecture flowchart](architecture.png)

The architecture flowchart is reproduced below as Mermaid source and included as `docs/architecture.png` in the submission zip.

```
User -> Streamlit UI
       -> Azure AI Vision 4.0 (Image Analysis: Caption + Tags) -> ingredients
       -> Azure AI Speech (Speech-to-Text)                      -> preferences
                                                                ↓
                                            Query Builder concatenates both
                                                                ↓
                          Azure OpenAI text-embedding-3-small (1536-d vector)
                                                                ↓
                                          FAISS IndexFlatIP (5,000 recipes)
                                                                ↓
                                                      Top-10 candidates
                                                                ↓
                              Azure OpenAI GPT-5-mini (JSON-mode rerank)
                                                                ↓
                                     3 ranked recipes + per-pick reasoning
                                                                ↓
                                                          Streamlit UI
```

## 4. Cloud Service Justification

Four Azure AI services drive the bot's logic. Each was chosen by weighing what the alternative would cost in latency, dollars, or complexity at the size of this demo.

**Azure AI Vision 4.0** (`ComputerVision` kind, F0 tier) provides the `analyze` API which returns a natural-language caption and a confidence-scored tag list in a single call. Tags are filtered against a food-noun whitelist (`src/vision.py:FOOD_TOKENS`) to recover ingredients. The alternative — training a Custom Vision model on a labelled fridge dataset — would have required hundreds of annotated images and produced a brittle classifier that fails on items outside the training distribution. The pretrained Vision 4.0 model generalises better and required zero training data.

**Azure AI Speech** handles voice-to-text via single-shot recognition in `australiaeast` (low latency from Melbourne). For 5–10 second utterances this is sufficient; the `recognize_once()` API is simpler than streaming and avoids the complexity of partial-result handling that streaming requires. Locale is set to `en-AU` so Australian pronunciation is handled accurately.

**Azure OpenAI `text-embedding-3-small`** produces 1536-dimensional embeddings. The same deployment is used at index-build time and at query time, ensuring the vector space is consistent. The alternative `text-embedding-3-large` (3072-d) would have doubled storage and per-query latency for marginal recall improvement on a 5,000-row corpus.

**Azure OpenAI GPT-5-mini** does the final rerank. The original plan called for `gpt-4o-mini`, but that model was deprecated by Microsoft on 2026-03-31 and cannot be deployed in any new Azure OpenAI resource as of the submission date. GPT-5-mini was selected as the cheapest currently-supported chat model with JSON-mode support. Two model-specific calling-convention changes were required: the parameter is `max_completion_tokens` (not `max_tokens`), and `temperature` cannot be overridden from the default value of 1.0. The recommender code documents both constraints inline (`src/recommender.py`).

**FAISS instead of Azure AI Search** is the one deliberate deviation from a pure-Azure stack, and it is documented as such. At 5,000 vectors, a flat inner-product index occupies ~30 MB on disk and answers queries in sub-millisecond time. Azure AI Search would add ~75 ms network round-trip and a $75/month minimum tier with no recall benefit at this scale. For a production deployment exceeding 100,000 vectors the calculus would flip — Azure AI Search's filtered hybrid retrieval becomes worth the cost — and the report acknowledges this in Section 8.

## 5. Recommendation Logic

The system prompt sent to GPT-5-mini is reproduced verbatim from `src/recommender.py`:

> *"You are a culinary assistant. Given a list of ingredients the user has on hand, their stated preferences, and 10 candidate recipes retrieved from a corpus, select the THREE best recipes for them. For each pick, give a one-sentence reason that references both the ingredients and the preferences. Respond ONLY as JSON of the form: `{"recommendations":[{"recipe_id":<int>,"name":"<str>","why":"<str>"}, ...]}`"*

Each candidate is formatted as a single line with its ID, name, cooking time, ingredient list (truncated to 10 items), and retrieval score. The model returns strict JSON; the application parses it with `json.loads` and truncates to the first three entries.

The three-stage funnel — Vision → Retrieval → LLM Rerank — beats either retrieval alone (cannot apply soft preferences like *"vegetarian"*) or LLM-alone (no grounding in real recipes, would hallucinate). Retrieval gives high recall over the corpus; the LLM reranks with reasoning the embedding cannot capture; every output is traceable back to a real `recipe_id`. A worked example with the inputs *(ingredients: tomato, garlic, basil, pasta, mozzarella; preferences: vegetarian, under 30 minutes)* returns:

```json
[
  {"recipe_id": 101, "name": "Pasta Marinara",
   "why": "Vegetarian and ready in 25 minutes, it uses your pasta, tomato, garlic and basil for a quick, fresh meal that fits your under-30-minute preference."},
  {"recipe_id": 103, "name": "Caprese Salad",
   "why": "A super-fast (~10 min) vegetarian option that highlights your tomato, mozzarella and basil for a simple, no-cook meal."},
  {"recipe_id": 106, "name": "Bruschetta",
   "why": "Quick (~15 min) and vegetarian, it makes great use of your tomato, garlic and basil for a light, speedy appetizer or meal."}
]
```

Note that the model correctly excluded `Pasta Carbonara` (contains bacon, not vegetarian) and `Beef Bourguignon` (180 minutes, exceeds time constraint), demonstrating that the soft-preference filtering is genuinely happening rather than the model simply picking the highest-scored candidates.

## 6. Implementation Screenshots

_Screenshots inserted from `docs/screens/` after live demo run:_

- `01-app-empty.png` — Streamlit landing state with both upload zones empty
- `02-uploaded.png` — Photo uploaded, ready to submit
- `03-vision-output.png` — Azure Vision status card expanded showing caption + tag list
- `04-recipes.png` — Three-card output with reasoning expanded
- `05-azure-resources.png` — `az resource list -g deakinuni -o table` proving Azure deployment under student account

## 7. Code Highlights

The implementation is approximately 250 lines of Python across six modules. Three highlights illustrate the architecture:

**`src/vision.py:analyze_image`** — One call to Azure AI Vision returns Caption + Tags. Tags are filtered against `FOOD_TOKENS` and `GENERIC_TAGS` to yield a clean ingredient list. Generic tags like `food` and `fridge` are explicitly excluded so the LLM is given specific ingredients rather than the word "food".

**`src/retrieval.py:RecipeRetriever.search`** — The query is embedded with the same deployment used at index-build time, L2-normalised, and matched against the FAISS index. The flat index gives exact cosine similarity; an approximate index (HNSW, IVF) was not warranted at 5,000 vectors.

**`src/recommender.py:recommend`** — The prompt builder, JSON-mode call to GPT-5-mini with `max_completion_tokens=2000`, and result parsing. The token budget is generous because GPT-5-mini consumes ~200 reasoning tokens even on trivial outputs (observed empirically during integration testing).

## 8. Limitations & Future Work

**Vision tag granularity.** The pretrained Vision 4.0 tags surface coarse categories (`fruit`, `vegetable`) more readily than specific ingredients (`kiwi`, `nectarine`). A domain-tuned Custom Vision model trained on a labelled fridge dataset would close this gap and is the natural next iteration. The current code path already accepts whatever ingredient list it is given, so swapping the extractor is a one-file change.

**Corpus locale.** Food.com is US-centric, which means the model occasionally recommends recipes with ingredients unfamiliar to an Australian shopper (e.g. *"Crisco"*, *"Velveeta"*). A multilingual or AU-localised dataset such as the *Allrecipes Australia* archive would improve relevance.

**No personalisation across sessions.** The current bot is stateless. Adding a user profile table and a feedback loop (thumbs-up / thumbs-down per recommendation) would let the system learn long-term preferences — favourite cuisines, allergies, disliked ingredients — and could be implemented as a Cosmos DB collection without changing the core flow.

**Scale.** At 5,000 vectors an in-memory FAISS index is correct. At >100,000 it would be worth migrating to Azure AI Search with hybrid (BM25 + vector) retrieval and pre-filterable metadata (cuisine, time, dietary tags), which would also enable hard constraints to be applied before the LLM rerank.

## 9. References (Harvard)

- Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré, P., Lomeli, M., Hosseini, L. & Jégou, H. 2024, *The Faiss library*, arXiv:2401.08281.
- FAO 2019, *The State of Food and Agriculture: Moving forward on food loss and waste reduction*, Food and Agriculture Organization of the United Nations, Rome.
- Li, S. 2019, *Food.com Recipes and Interactions* [dataset], Kaggle, viewed 24 May 2026, <https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions>.
- Microsoft 2024, *Azure AI Vision 4.0 — Image Analysis*, Microsoft Learn, viewed 24 May 2026, <https://learn.microsoft.com/azure/ai-services/computer-vision/concept-tag-images-40>.
- Microsoft 2024, *Azure AI Speech Service*, Microsoft Learn, viewed 24 May 2026, <https://learn.microsoft.com/azure/ai-services/speech-service/>.
- Microsoft 2024, *Azure OpenAI Service*, Microsoft Learn, viewed 24 May 2026, <https://learn.microsoft.com/azure/ai-services/openai/>.
- OpenAI 2024, *Embeddings API reference*, viewed 24 May 2026, <https://platform.openai.com/docs/api-reference/embeddings>.
