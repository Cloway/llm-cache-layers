# AGENTS.md

Instructions for coding agents working on this repository.

## What it is

A benchmark of caching strategies for LLM applications. `lab.py` builds question workloads, runs cache scenarios against an OpenAI-compatible API and writes one row per request. `analyze.py` turns the results into tables and charts.

The questions, the store rules and the classifier prompt are in Brazilian Portuguese. The published results depend on that text. Do not translate or edit `data/` unless you intend to rerun every scenario.

## Setup

- Python 3.10 or newer. `lab.py` uses only the standard library.
- `pip install -r requirements.txt` only to run `analyze.py`.
- Copy `.env.example` to `.env` and fill in `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `CHAT_DEPLOYMENT` and `EMBEDDINGS_DEPLOYMENT`.

## Rules

- Never print, log or commit the contents of `.env`.
- `run` and `calibrate` make paid API calls. Run `simulate` with the same arguments first and show the worst case to whoever asked for the run.
- Keep `--max-calls`. It stops the run when the number of API calls goes over the limit.
- `analyze.py`, `simulate` and `project` make no API calls.
- Do not delete or rewrite files in `results/`. Each run creates new files, named with timestamp, workload size, FAQ share, scenarios and threshold.
- If you change prices, edit `prices.json` and record the source and date in the file.

## Commands

```bash
python lab.py simulate  --n 100 --faq-share 0.5 --scenarios ABCDEFG   # no API calls
python lab.py calibrate                                               # 70 embedding calls the first time
python lab.py run       --n 100 --faq-share 0.5 --threshold 0.80 --scenarios ABCDE
python lab.py run       --n 300 --faq-share 0.5 --threshold 0.80 --scenarios G --max-calls 950
python lab.py project   --n 1000                                      # no API calls
python analyze.py                                                     # no API calls
```

| Option | Meaning | Default |
|---|---|---|
| `--n` | requests in the workload | 30 |
| `--faq-share` | share of frequent questions among non-personal requests | 0.5 |
| `--personal-share` | share of requests that depend on customer data | 0.2 |
| `--threshold` | minimum cosine similarity for the semantic cache | 0.75 |
| `--seed` | workload seed | 7 |
| `--scenarios` | letters of the scenarios to run | ABCDE |
| `--k` | confirmations from different users to approve a group (F, G) | 2 |
| `--audit-rate` | share of approved-group hits that still call the model (F, G) | 0.1 |
| `--max-calls` | hard limit of API calls in the run | 250 |

## Scenarios

| Letter | Layers |
|---|---|
| A | none (a random prefix defeats the prompt cache) |
| B | prompt cache |
| C | B + exact cache |
| D | C + semantic cache for every question |
| E | C + semantic cache filtered by an LLM classifier |
| F | C + semantic cache with a per-group status in SQLite |
| G | F + rule: a question that contains a digit skips the semantic cache |

A to E are in `run_scenario`. F and G are in `run_status_scenario`.

## Result files

`results/<timestamp>-n<N>-faq<P>-<scenarios>-thr<T>-rows.jsonl`, one JSON object per request:

| Field | Content |
|---|---|
| `scenario`, `i` | scenario letter and position in the workload |
| `type` | `FAQ`, `UNIQUE` or `PERSONAL` |
| `group` | question id; for personal questions it includes the customer |
| `answer_key` | expected answer; two requests with the same key can share an answer |
| `question`, `answer` | text sent and text returned |
| `source` | `model`, `exact` or `semantic` |
| `correct` | `true` or `false` when the answer came from a cache; `null` when it came from the model |
| `served_answer_key` | answer key of the stored answer, on semantic hits |
| `input_tokens`, `cached_tokens`, `output_tokens` | tokens of the main call (`cached_tokens` from the API usage) |
| `clf_input_tokens`, `clf_output_tokens`, `clf_label` | classifier call (scenario E) |
| `embedding_tokens` | embedding tokens charged to the request |
| `similarity` | similarity to the nearest cache entry |
| `group_status` | `NEW`, `OBSERVING`, `APPROVED`, `BLOCKED` or `RULE` (F, G) |
| `audit` | the request was an audit of an approved group (F, G) |
| `latency_s` | seconds summed over every call of the request |
| `cost_usd` | dollars charged to the request |

The `-summary.json` file with the same prefix has the parameters and the totals per scenario. The `.sqlite` files of scenarios F and G have a `groups` table with the final status of each group. `results/embeddings.json` caches the vectors of the fixed phrasings; `analyze.py` needs it.

The stored results were produced with Portuguese field names and converted to the names above. Every value is unchanged except `correct` on semantic hits of the 100-request runs, which was recomputed from the answer key, as the analysis always did.

## Extending

- **Another model:** set `CHAT_DEPLOYMENT`, update `prices.json` and run again. Compare `cost_per_1000_usd` and the `cached_tokens / input_tokens` share.
- **Another application:** replace `data/system_prompt.md` and `data/questions.json`. The prompt cache needs at least 1,024 identical tokens at the start of the prompt.
- **New scenario:** add the letter to `run_scenario` or write a function like `run_status_scenario`, add its worst case to `cmd_simulate` and its dispatch to `cmd_run`.
- **Answer key:** if you add unique-question templates, update `unique_answer_key`, which defines when two questions can share an answer.
