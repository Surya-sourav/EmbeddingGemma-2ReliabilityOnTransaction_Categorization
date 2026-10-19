# How reliable is EmbeddingGemma 2 at knowing when it’s wrong about a bank transaction?

A small, reproducible experiment on real card transactions. The short answer: it's excellent at recognising
merchants it has seen before, but it gets about half of new merchants get wrong, and its the similarity score can't tell
us that which answers to trust at the end.

---

## The Blind spot (Q1)

I build production AI systems at a fintech startup, and a big part of that is categorising card transactions.
The usual approach is embedding retrieval and matching it with the close aly. when we embed the raw descriptor (`SQ *BLUE BOTTLE`,
`AMZN MKTP US*Z58J27N20`), we tend to find the most similar transaction that we've already labelled, and copy its category. [ Many Fintech companies across the world utilizes the similar procedure]

however , On a dashboard this looks great, because most traffic comes from merchants we've seen many times. In an
earlier study of my own, accuracy was 0.93–0.98 for merchants that were seen before but only **0.599 for merchants
that were never seen before**, and the confidence signals or any benchmarks also didn't flag the wrong answers.

That's the blind spot: **for merchant names, embedding similarity measures "have I seen something like this
string?", not the question about "is this categorisation right?"** Standard embedding benchmarks don't catch this, because
they score models on clean sentences and paragraphs in a fixed collection and to be honest Transaction descriptors are
different:
- They're short and can be in any form (basically high variability )(median 22 characters here), upper-case, truncated name fragments.
- They're full of processor prefixes, store numbers and order codes.
- The set of merchants keeps changing: every week brings merchants the system has never seen.

I have tested across five hypotheses:

| | Hypothesis & Testing |
|---|---|
| H1 | Accuracy doesn't decrease gradually as merchants get rarer; it falls drastically for an unseen merchant. |
| H2 | A high similarity score doesn't reliably mean that the match is right. |
| H3 | Different merchants with lookalike names get high similarity ("false friends / false positves"). |
| H4 | The redundant noise (prefixes, store numbers, truncation) affects the matching process a lot. |
| H5 | Whether a simple character n-gram matcher is about as good, because descriptors behave like names, not prose. |

## Why this model

[`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) is a current open-weight embedding
model from Google. It's the kind of model a fintech team would reach for today for retrieval-based
categorisation:
- **Licence and access:** Apache 2.0, and not gated.
- **Size:** 740M parameters, so it fits the 0.6–6B range and runs on a free Colab T4. Text goes through the
  roughly 270M-parameter text encoder; the vision and audio parts gets loaded but is unused.
- **Task prompts:** it ships with documented task prompts, so I could test it exactly the way its authors
  say to use it ( as per the model card ).
- **Setup used:** full checkpoint, pinned to revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, in
  float32 (the model card warns against float16).

## Dataset and licence

**Source:** [State of Delaware: State Employee Credit Card Transactions](https://data.delaware.gov/d/nurt-5rqw).
These are the real card transactions, each with a date, the raw merchant descriptor as printed by the card network or operator, and
the merchant category the card network has on file (the MCC description).

**Licence:** Public Domain (State of Delaware Open Data; attribution: Department of Finance, Division of
Accounting). That's why the evaluation data is included in this bucket under `data/`.

**What I used:**
- Transactions from July 2022 to June 2026 (four fiscal years): **762,506 transactions**, 189,235 different
  descriptor strings, 250 categories.
- I pre-processed the category text, so "STATIONERY-OFFICE SUPPLIES" and "Stationery Office Supplies" count as the same.
- I dropped internal accounting rows such as chargebacks ( not necessary as per our hypothesis provenance ).

**Merchant names.** The data has no merchant ID, so I derived a merchant name with a few readable rules & they are:
- drop the processor prefix (`SQ *`, `TST*`, `PAYPAL *`, `SP `…);
- cut at the first store or order number (`WAL-MART #2791` → `WALMART`, `STAPLS0239279988000001` → `STAPLS`);
- drop punctuation and words like INC/LLC/THE.

That gives exact 45,396 merchants. The rule isn't perfect but it keeps `EASTERN SHORE COFFEE WA` and
`EASTERN SHORE COFFEE W` apart. That's why accuracy is judged on the **category**, which comes from the card
network. The merchant name is only used to decide whether a merchant was seen before or not.

## The Methodology:

Think of a cashier with a notebook of every past receipt. A new receipt comes in, they find the line that
looks most like it, and they copy its category. That's embedding retrieval, and that's what I have measured.

1. **Past vs. the future.** Everything before **4 Sep 2025** (80% of the transactions) is the "notebook": 609,394
   transactions, reduced to 154,361 unique descriptors. Each keeps the category it had most often. From the
   153,112 later transactions I randomly sample **20,000 queries** (seed 42).
2. **How familiar is each merchant?** For each query that I count, how often its merchant appears in the past ( frequency ):
   - 0 times: 1,960 queries
   - once: 335
   - 2–5 times: 848
   - 6+ times: 16,857

   About 7 in 10 later transactions are an exact repeat of a descriptor already in the notebook.
3. **Lookup.** Queries are embedded with the model's `query` prompt and past descriptors with its `document`
   prompt. For each query I take the single closest past descriptor (cosine similarity) and copy its category.
4. **Baseline.** I do the same lookup with character 3–5-gram TF-IDF, which uses no neural network.
5. **Checks:**
   - accuracy per familiarity bucket, with 95% margins;
   - how similarity differs between right and wrong answers (AUROC);
   - accuracy per similarity band, and what a trust threshold would let through;
   - whether a low score flags new merchants;
   - the most confident mistakes;
   - 1,000 descriptors corrupted with prefixes, store numbers and truncation;
   - a re-run without prompts and with the STS prompt, to make sure the prompt isn't the cause.

## Outcomes : 

All numbers come from the notebook that ran on a Colab T4.

### H1: it's a cliff, not a slope (confirmed)

| Times the merchant was seen before | Queries | EmbeddingGemma 2 | TF-IDF baseline | Found the same merchant |
|---|---|---|---|---|
| 0 (never) | 1,960 | **50.9% ± 2.2** | 43.1% | n/a |
| 1 | 335 | 92.2% ± 2.9 | 91.9% | 97.0% |
| 2–5 | 848 | 92.1% ± 1.8 | 92.0% | 96.6% |
| 6+ | 16,857 | 95.9% ± 0.3 | 95.4% | 99.3% |
| all | 20,000 | 91.2% ± 0.4 | 90.1% | 89.4% |

*Values are the share of queries given the right category, with 95% margins. Always guessing the most common
category ("book stores") gets 11.8%.*

Seeing a merchant once is almost as good as seeing it hundreds of times (92% vs 96%). Never having seen it
halves the accuracy. The overall 91% hides this completely, because new merchants are only about 10% of the
traffic.

![Accuracy by times seen](results/accuracy_by_times_seen.png)

### H2: a high score doesn't mean a right answer (confirmed)

| Times seen | Avg similarity, right answers | Avg similarity, wrong answers | AUROC (score separates right from wrong) |
|---|---|---|---|
| 0 | 0.879 | 0.839 | 0.75 |
| 1 | 0.960 | 0.946 | 0.58 |
| 2–5 | 0.959 | 0.933 | 0.69 |
| 6+ | 0.955 | **0.954** | **0.41** |
| all | 0.951 | 0.890 | 0.72 |

*An AUROC of 0.5 is a coin flip; 1.0 is perfect.*

- For well-known merchants, wrong answers score just as high as right ones, so the score is worse than a coin
  flip there.
- Accuracy stops improving above 0.90:

  | Similarity band | Accuracy | Queries |
  |---|---|---|
  | 0.75–0.80 | 19.0% | 153 |
  | 0.80–0.85 | 35.3% | 834 |
  | 0.85–0.90 | 68.3% | 967 |
  | 0.90–0.95 | 96.4% | 4,656 |
  | 0.95–1.00 | 95.4% | 13,389 |

- A trust threshold doesn't fix it:

  | Trust if similarity ≥ | Queries kept | Accuracy of kept | Wrong answers still kept |
  |---|---|---|---|
  | 0.80 | 99.2% | 91.8% | 1,627 |
  | 0.90 | 90.2% | 95.7% | 780 |
  | 0.95 | 66.9% | 95.4% | 614 |

**The twist:** the same score is very good at a different question. Its like as an alarm for *"this merchant is new"*,
low similarity reaches **AUROC 0.97** (TF-IDF: 0.81). So the score knows when it's in unfamiliar territory;
it just doesn't know when it's wrong.

![Similarity of right vs wrong answers](results/similarity_right_vs_wrong.png)
![Accuracy by similarity](results/accuracy_by_similarity.png)

### H3: lookalikes (partly)

The highest-scoring matches to a *different* merchant name (0.94–0.97) are mostly the same business written
differently: `TST* ILLY S COFFEE- PHL A` vs `TST* ILLY'S COFFEE- PHL A`, `SAMS CLUB.COM` vs `SAMSCLUB.COM`.
Real false positives that exist (`SP WORLD OF BOOKS` → `SP WORLD OF READING`), but they're rare at the top.

The confident mistakes are mostly something else: **one name that carries more than one category** (please see the
next section).

### H4: descriptor noise (not really)

| Noise added to 1,000 known descriptors | Embedding finds the merchant | TF-IDF finds the merchant | Embedding right category | Avg similarity |
|---|---|---|---|---|
| none | 100.0% | 100.0% | 99.9% | 0.966 |
| cut to 12 characters | 91.3% | 91.1% | 96.1% | 0.933 |
| add `SQ *` | 99.7% | 99.9% | 99.6% | 0.925 |
| add `TST* ` | 99.3% | 100.0% | 99.6% | 0.921 |
| add `PAYPAL *` | 97.2% | 99.9% | 96.9% | 0.891 |
| add a store number (` #0412`) | 99.8% | 100.0% | 99.7% | 0.924 |
| prefix + store number | 97.4% | 100.0% | 97.7% | 0.882 |

The model handles everyday noise well. Truncation hurts most, and plain n-grams cope just as well or better.
What noise does change is the **score**: it drops by up to 0.08, so a threshold tuned on clean descriptors
would start rejecting correct matches.

### H5: n-grams are as good on known merchants, but not on new ones

On merchants seen before, TF-IDF is within half a point of the model. On never-seen merchants the model is
clearly better (50.9% vs 43.1%), and it's much better at noticing novelty (AUROC 0.97 vs 0.81). So the
model's real added value is on exactly the cases where it's least reliable.

### It isn't the prompt

| Prompt setup | All queries | Never-seen merchants |
|---|---|---|
| documented (`query` / `document`) | 91.2% | 50.9% |
| no prompt | 91.5% | 53.0% |
| `STS` prompt on both sides | 91.2% | 50.5% |

## Example failures

The most confident wrong answers (full lists in `results/wrong_answers.csv` and `results/lookalikes.csv`):

| Seen before | New transaction | Closest past descriptor | Similarity | True category | Copied category |
|---|---|---|---|---|---|
| 1× | `FAIRMONT BREAKERS LONG B` | `FAIRMONT BREAKERS LONG B` | 0.976 | lodging | eating places / restaurants |
| 1× | `EXXON CRYSTAL HILL MAR` | `EXXON CRYSTAL HILL MAR` | 0.974 | fuel dispenser, automated | service stations |
| 1× | `PHILLIPS 66 - SAMI'S` | `PHILLIPS 66 - SAMI'S` | 0.972 | service stations | fuel dispenser, automated |
| never | `TST*HALF-MOON EMPANADAS` | `TST*HALF MOON EMPANADAS` | 0.965 | eating places / restaurants | fast food restaurants |
| never | `DELAWARE MUSEUM OF NATUR` | `DELAWARE MUSEUM OF NATURA` | 0.962 | tourist attractions | gift / souvenir shops |
| never | `GO-GLASS` | `GO GLASS` | 0.955 | automotive service shops | special trade contractors |
| never | `SP WORLD OF BOOKS` | `SP WORLD OF READING` | 0.938 | book stores | specialty retail |
| never | `SHERATON GRAND SEATTLE` | `SHERATON SEATTLE` | 0.935 | lodging | eating places / restaurants |

The pattern is clear. A hotel and its restaurant, a museum and its gift shop, and a gas station and its pump
share one name but have different categories. The model finds the right *name* with very high similarity and
copies the wrong *category*. Several of these are the exact same string in the past and the future. No
similarity threshold can catch them, and that's a big reason the score fails as a confidence signal.

## Analysis (Q3)

Speculative, but grounded in what the numbers show:

1. **Using similarity for routing,but not for confidence.** It's a strong novelty detector (AUROC 0.97). by sending the 
   low-similarity transactions, which are mostly new merchants, to a different path: an LLM with merchant
   enrichment, a web lookup, or human review would help in categorizing accuratel but also Keeping the lookup only for merchants that have been seen before.
2. **Learning the confidence instead of reading it off the cosine.** A small calibrated classifier trained on
   past outcomes could use the similarity, the gap between the top two matches, how often the merchant has
   been seen, the TF-IDF score, and whether the embedding and TF-IDF agree. That would give a confidence
   that means something or maybe correctness .
3. **Curated data for the cases that fail, and then we may fine-tune.**
   - Mine (descriptor variant → merchant) pairs from history; `AMZN MKTP US*…` alone has 35,000 variants. Use them for
     contrastive fine-tuning, with lookalikes as hard negatives.
   - More importantly, train descriptor → category directly, so a new merchant can be categorised from what
     its name says ("EMPANADAS" → restaurant) instead of by copying the nearest neighbour.
4. **Fixing the label side.** Many confident errors are one merchant with several valid categories. by storing a
   category distribution per merchant, and use simple context such as the amount, or the time of day, to pick
   between "hotel" and "hotel restaurant".
5. **Hybrid retrieval.** TF-IDF matches the model on known merchants and copes better with prefixes. Fusing
   the lexical and dense scores is cheap and removes some failure modes effectively.

## Reproducibility

**Run it:**
1. Open `transactions_gemma2Embeddings_blindspot.ipynb` in Google Colab.
2. Choose *Runtime → Change runtime type → T4 GPU*, then *Run all*. It takes about 35 minutes
3. No Hugging Face token is needed: the model isn't gated, and the data downloads directly from the Delaware
   portal.
4. The last cell downloads `results_and_data.zip.

**Pinned:**
- `sentence-transformers==6.1.0`, `transformers==5.20.1`, `huggingface_hub==1.33.0`
- model revision `914f7f89142e33e77833254d9c9b90c3cef7303b`
- split date 2025-09-04, seed 42 (query sample and noise sample)
- torch, torchvision, scikit-learn, pandas and matplotlib are Colab's preinstalled versions

The data window is fixed (Jul 2022 – Jun 2026), so later updates to the portal don't change the experiment.

Author : Surya Parida 
Full-fillment for FIT 2026


