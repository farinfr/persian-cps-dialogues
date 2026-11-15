# Persian Conversational Product Search Dialogues

Code and data for the paper **"Automatic Construction of E-Commerce Dialogue Datasets for Conversational Search"** (Naderi & Nafar, Iran University of Science and Technology).

The framework builds a Persian conversational product-search (CPS) dataset from a real e-commerce catalog:

1. **Crawl** 12,926 products in 7 categories of [Digikala](https://www.digikala.com) (`legacy/crawler`).
2. **Cluster** them into 170 semantically coherent clusters (K-Means on multilingual MiniLM embeddings, `legacy/clustering`, `legacy/embedding`).
3. **Plan the questions** of each dialogue with the corrected **Regularized Partition Balance (RPB)** score, recomputed on the system's belief after every answer (`cpsdata/rpb.py`, `cpsdata/scoring.py`).
4. **Augment** each product with comparisons to its 10 nearest neighbours (price position, rating, popularity; `cpsdata/augment.py`).
5. **Verbalize** the plan into a requirement-driven Persian dialogue with the free Google Gemini API (Single Prompt Strategy, `scripts/generate_dialogues.py`).

## Contents

```
cpsdata/                 library: catalog loading, feature tables, RPB score, augmentation, Gemini client
scripts/
  build_catalog.py               raw crawl -> data/catalog.jsonl.gz, data/embeddings.npz
  simulate_question_selection.py Section 4.3: RPB vs. entropy vs. initial score with a simulated user
  check_initial_definition.py    Section 3.3: pairwise check of the initial score D(P)
  generate_dialogues.py          builds the dataset (resumable, respects the free-tier quota)
  judge_dialogues.py             LLM-as-judge scores (Table 7 criteria)
  dataset_stats.py               statistics and automatic consistency checks
data/
  catalog.jsonl.gz       12,926 products with flattened specifications and cluster ids
  embeddings.npz         MiniLM embeddings (float16) used for clustering and augmentation
dataset/
  dialogues.jsonl        the released dialogues (one JSON object per line)
results/                 outputs of the scripts (simulation, judge, statistics)
legacy/                  code and results of the first version (crawler, clustering, method comparison
                         with Qwen-Turbo / Llama 3 / GPT-4o, the 26 dialogues of the first dataset)
tests/test_rpb.py        properties of the RPB score; vectorised vs. reference implementation
```

## Quick start

```bash
pip install -r requirements.txt

# reproduce the simulation of Section 4.3 (about 15 minutes on a laptop CPU, no API needed)
python scripts/simulate_question_selection.py --out results/simulation

# reproduce the pairwise check of the initial score (Section 3.3)
python scripts/check_initial_definition.py

# tests
python -m pytest tests
```

### Generating dialogues

Create a free key at <https://aistudio.google.com/apikey> and put it in `.env`:

```bash
cp .env.example .env   # then edit: GEMINI_API_KEY=...
python scripts/generate_dialogues.py --dry-run        # print one prompt, no API call
python scripts/generate_dialogues.py --n 1000         # resumable; run again after the daily quota resets
python scripts/judge_dialogues.py --sample 100 --legacy legacy/v1_results/final_dataset_v1/llm_generated_dialogues_highest_metric.json
python scripts/dataset_stats.py
```

The generator tries `gemini-3.5-flash` first and moves down a list of free models (`gemini-3-flash-preview`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, …) when the free daily quota of a model is exhausted; each record stores the model that produced it. The released 1,000 dialogues were produced over two days by `gemini-3.1-flash-lite` (491), `gemini-3.5-flash-lite` (455), `gemini-3-flash-preview` (34) and `gemini-3.5-flash` (20). Use `--models` to choose others and `--interval` to change the pause between requests. Outputs that fail validation are saved in `results/failed/`; `--recover-failed` reuses the valid ones instead of calling the API again.

## Dataset format

Each line of `dataset/dialogues.jsonl` is one dialogue:

| field | description |
|---|---|
| `id`, `product_id`, `digikala_id` | identifiers of the target product (`digikala_id` = `dkp-…` on digikala.com) |
| `cluster`, `category`, `product_type`, `title`, `price` | target product (price in rial) |
| `question_plan` | ordered list of `{feature, value, rpb, numeric_bin}`; `value: null` = the user does not know / does not care |
| `target_rank_after_plan`, `effective_candidates_after_plan` | rank of the target and effective number of candidates after the planned answers |
| `baseline_initial_rpb` | the five features the first dataset version would have used, for comparison |
| `candidate_product_ids` | target followed by the two alternatives presented in the dialogue |
| `augmentation_labels` | comparison labels of the target (e.g. `good_price`, `popular_choice`) |
| `dialogue` | list of `{role: user\|assistant, text, feature}`; `feature` tags assistant turns that ask about a planned feature |
| `summary`, `model`, `prompt_version`, `created_at` | generation metadata |

## The RPB score

For feature *P*, belief *b* over the cluster's products, coverage κ = belief mass of products that have *P*, value shares *p_i* among them, effective number of candidates *N_P* (Kish) and *m_P* non-negligible values:

```
RPB(P | b) = (1 − δ̂_P) · κ_P · [ H(P) − α · max(0, log2 m_P − ½ log2 N_P) ]₊
```

With full coverage, no "don't know" answers and α = 0 it equals the entropy (information gain) of the question. See Section 3.3 of the paper and `cpsdata/rpb.py`.

## Licence and terms of use

* Code: MIT licence (`LICENSE`).
* Product data are taken from public product pages of Digikala and remain subject to Digikala's terms; the catalog and the dialogues are shared **for non-commercial research purposes only**.
* The dialogues were generated by an LLM and may contain errors; they are not real customer conversations.

## Citation

```bibtex
@article{naderi2026persiancps,
  title  = {Automatic Construction of E-Commerce Dialogue Datasets for Conversational Search},
  author = {Naderi, Hassan and Nafar, Fariba},
  year   = {2026},
  note   = {Under review}
}
```
