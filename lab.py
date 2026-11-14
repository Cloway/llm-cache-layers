"""LLM cache layers benchmark. Standard library only.

Commands:
  python lab.py calibrate            similarity between the fixed questions, per threshold
  python lab.py simulate --n 30      worst-case calls and cost, no API calls
  python lab.py run      --n 30      run the scenarios and write the results
  python lab.py project  --n 1000    project scenarios A, B, C to a larger volume, no API calls

Scenarios:
  A  no cache: a random identifier at the start of the system prompt defeats prompt caching
  B  prompt cache: identical system prompt on every call, question at the end
  C  B + exact cache (hash of the question and the customer data)
  D  C + semantic cache for every question
  E  C + semantic cache only for questions an LLM classifier labels as reusable
  F  C + semantic cache with a per-group status kept in SQLite
  G  F + rule: a question that contains a digit never enters the semantic cache

The questions, the store rules and the classifier prompt are in Brazilian Portuguese on purpose.
The published results depend on that text, so it is data and stays untranslated.
"""
import argparse
import hashlib
import json
import random
import re
import sqlite3
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
MAX_OUTPUT_TOKENS = 200
ANSWER_MATCH_THRESHOLD = 0.80  # scenarios F and G: similarity required between two answers

# Classifier prompt for scenario E (Portuguese, part of the experiment).
CLASSIFIER_PROMPT = (
    "Voce decide se a resposta a uma pergunta de cliente de loja pode ser guardada e reutilizada "
    "para outras pessoas. Responda com uma palavra: REUTILIZAVEL ou NAO.\n"
    "REUTILIZAVEL: pergunta generica sobre regras, prazos ou politicas, sem valores nem caso especifico.\n"
    "NAO: a resposta depende dos dados de quem pergunta (o pedido, o saldo, a entrega dele) ou de "
    "detalhes do caso citados na pergunta (um valor em reais, uma cidade, uma quantidade de dias, um produto)."
)
EXPECTED_LABEL = {"FAQ": "REUSABLE", "UNIQUE": "NOT_REUSABLE", "PERSONAL": "NOT_REUSABLE"}
CITIES = ["Recife", "Curitiba", "Manaus", "Belo Horizonte", "Campinas", "Santos", "Juazeiro do Norte",
          "Londrina", "Porto Alegre", "Sorocaba", "Salvador", "Uberaba"]
STATE_CAPITALS = {"Recife", "Curitiba", "Manaus", "Belo Horizonte", "Porto Alegre", "Salvador"}
PRODUCTS = ["fone de ouvido", "sofa", "tenis", "notebook", "guarda-roupa", "camisa", "televisor", "mesa de jantar"]
WARRANTY_MONTHS = {"fone de ouvido": 12, "notebook": 12, "televisor": 12, "sofa": 6, "guarda-roupa": 6,
                   "mesa de jantar": 6, "tenis": 3, "camisa": 3}


# ---------- configuration ----------

def read_env():
    cfg = {}
    if not (ROOT / ".env").exists():
        return cfg  # simulate, project and analyze.py work without credentials
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        m = re.match(r"\s*([A-Z_]+)=(.*)", line)
        if m:
            cfg[m.group(1)] = m.group(2).strip()
    return cfg


ENV = read_env()
BASE_URL = ENV.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
CHAT_MODEL = ENV.get("CHAT_DEPLOYMENT") or "gpt-5.4-nano"
EMBEDDINGS_MODEL = ENV.get("EMBEDDINGS_DEPLOYMENT") or "text-embedding-3-small"
PRICES = json.loads((ROOT / "prices.json").read_text(encoding="utf-8"))
SYSTEM_PROMPT = (ROOT / "data" / "system_prompt.md").read_text(encoding="utf-8")
DATA = json.loads((ROOT / "data" / "questions.json").read_text(encoding="utf-8"))


# ---------- API calls ----------

class CallBudget:
    made = 0
    maximum = 0


def post(path, body):
    if not BASE_URL or not ENV.get("AZURE_OPENAI_API_KEY"):
        sys.exit("Create a .env file from .env.example and fill in the endpoint and the key.")
    if CallBudget.made >= CallBudget.maximum:
        sys.exit(f"Call limit of {CallBudget.maximum} reached. Run stopped.")
    CallBudget.made += 1
    req = urllib.request.Request(
        f"{BASE_URL}/{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"api-key": ENV["AZURE_OPENAI_API_KEY"], "Content-Type": "application/json"},
    )
    for attempt in range(5):
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read()), time.perf_counter() - t0
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 4:
                wait = float(e.headers.get("Retry-After") or 5 * (attempt + 1))
                print(f"    HTTP {e.code}, waiting {wait:.0f}s")
                time.sleep(min(wait, 60))
                continue
            sys.exit(f"HTTP error {e.code}: {e.read().decode('utf-8', 'replace')[:400]}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == 4:
                sys.exit(f"Network failure after 5 attempts: {e}")
            print(f"    network: {e}, retrying in {10 * (attempt + 1)}s")
            time.sleep(10 * (attempt + 1))


def chat(system, user, max_output_tokens=MAX_OUTPUT_TOKENS):
    r, latency = post("chat/completions", {
        "model": CHAT_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_completion_tokens": max_output_tokens,
    })
    u = r["usage"]
    return {
        "text": (r["choices"][0]["message"]["content"] or "").strip(),
        "input_tokens": u["prompt_tokens"],
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
        "output_tokens": u["completion_tokens"],
        "latency_s": latency,
    }


def load_embeddings():
    path = RESULTS / "embeddings.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


EMBEDDINGS = {}


def embed(text):
    """Embedding of a text. Kept on disk so repeated runs do not call the API again;
    the cost and latency measured the first time are charged on every request."""
    if text not in EMBEDDINGS:
        r, latency = post("embeddings", {"model": EMBEDDINGS_MODEL, "input": text})
        EMBEDDINGS[text] = {"vector": r["data"][0]["embedding"], "tokens": r["usage"]["prompt_tokens"],
                            "latency_s": latency}
    return EMBEDDINGS[text]


def save_embeddings():
    """Write the vectors of the fixed questions to disk (unique questions do not repeat across runs)."""
    fixed = {p for block in ("faq", "personal") for g in DATA[block] for p in g["phrasings"]}
    on_disk = load_embeddings()
    on_disk.update({k: v for k, v in EMBEDDINGS.items() if k in fixed})
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "embeddings.json").write_text(json.dumps(on_disk), encoding="utf-8")


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))  # the model returns unit vectors


# ---------- cost ----------

def chat_cost(input_tokens, cached_tokens, output_tokens):
    p = PRICES["chat"]
    return ((input_tokens - cached_tokens) * p["input"] + cached_tokens * p["cached_input"]
            + output_tokens * p["output"]) / 1e6


def embedding_cost(tokens):
    return tokens * PRICES["embeddings"]["input"] / 1e6


def row_cost_parts(row):
    p = PRICES["chat"]
    return {
        "uncached input": (row["input_tokens"] - row["cached_tokens"]) * p["input"] / 1e6,
        "cached input": row["cached_tokens"] * p["cached_input"] / 1e6,
        "output": row["output_tokens"] * p["output"] / 1e6,
        "classifier": chat_cost(row["clf_input_tokens"], 0, row["clf_output_tokens"]),
        "embeddings": embedding_cost(row["embedding_tokens"]),
    }


# ---------- workload ----------

def normalize(text):
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text)).strip()


def unique_question(rnd):
    """A question with its own values, which does not repeat (Portuguese templates)."""
    v = rnd.randrange(40, 1200)
    return rnd.choice([
        f"Um pedido de R$ {v} para {rnd.choice(CITIES)} tem frete gratis?",
        f"Quero parcelar uma compra de R$ {v} em {rnd.randrange(2, 13)} vezes. Pode?",
        f"Quantos pontos ganho em uma compra de R$ {v}?",
        f"Recebi meu pedido ha {rnd.randrange(2, 60)} dias. Ainda posso trocar?",
        f"Comprei um {rnd.choice(PRODUCTS)} ha {rnd.randrange(1, 18)} meses e deu defeito. Ainda esta na garantia?",
    ])


def unique_answer_key(q):
    """Expected answer of a unique question, computed from the store rules.
    Two questions with the same key can share an answer without error.
    Keys: frete = free shipping, parcelas = installments, pontos = points, troca = exchange, garantia = warranty."""
    n = [int(x) for x in re.findall(r"\d+", q)]
    if "frete" in q:
        return f"frete:{n[0] > 199 and any(c in q for c in STATE_CAPITALS)}"
    if "parcelar" in q:
        return f"parcelas:{n[1] <= 10 and n[0] / n[1] >= 30}"
    if "pontos" in q:
        return f"pontos:{n[0]}"
    if "trocar" in q:
        return f"troca:{n[0] <= 30}"
    product = next(p for p in WARRANTY_MONTHS if p in q)
    return f"garantia:{n[0] <= WARRANTY_MONTHS[product]}"


def answer_key(group):
    return unique_answer_key(group[7:]) if group.startswith("unique:") else group


def build_workload(n, faq_share, personal_share, seed):
    """Sequence of n requests with three kinds of question.
    PERSONAL (share `personal_share`): depends on the customer data.
    FAQ (share `faq_share` of the rest): one of 16 frequent questions, in one of 4 phrasings.
    UNIQUE (what is left): a question with its own values, which does not repeat."""
    rnd = random.Random(seed)
    used, workload = set(), []
    for _ in range(n):
        if rnd.random() < personal_share:
            p = rnd.choice(DATA["personal"])
            c = rnd.choice(DATA["customers"])
            workload.append({"type": "PERSONAL", "group": f"{p['group']}:{c['id']}",
                             "question": rnd.choice(p["phrasings"]), "context": c["data"]})
        elif rnd.random() < faq_share:
            g = rnd.choice(DATA["faq"])
            workload.append({"type": "FAQ", "group": g["group"], "question": rnd.choice(g["phrasings"]), "context": ""})
        else:
            q = unique_question(rnd)
            while q in used:
                q = unique_question(rnd)
            used.add(q)
            workload.append({"type": "UNIQUE", "group": f"unique:{q}", "question": q, "context": ""})
    rnd_users = random.Random(seed + 1000)
    for item in workload:
        item["answer_key"] = answer_key(item["group"])
        item["user"] = item["group"].split(":")[1] if item["type"] == "PERSONAL" else f"u{rnd_users.randrange(60)}"
    return workload


def user_message(item):
    if item["context"]:
        return f"Dados do cliente: {item['context']}\n\nPergunta: {item['question']}"
    return f"Pergunta: {item['question']}"


# ---------- scenarios ----------

def new_row(scenario, i, item):
    return {"scenario": scenario, "i": i, "type": item["type"], "group": item["group"],
            "answer_key": item["answer_key"], "question": item["question"], "source": "model", "correct": None,
            "served_answer_key": None, "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
            "embedding_tokens": 0, "clf_input_tokens": 0, "clf_output_tokens": 0, "clf_label": None,
            "similarity": None, "latency_s": 0.0, "group_status": None, "audit": False}


def record_model_call(row, r):
    row["input_tokens"], row["cached_tokens"], row["output_tokens"] = r["input_tokens"], r["cached_tokens"], r["output_tokens"]
    row["latency_s"] += r["latency_s"]


def finish_row(row, answer, rows):
    row["answer"] = answer
    row["cost_usd"] = sum(row_cost_parts(row).values())
    rows.append(row)
    mark = {"model": ".", "exact": "=", "semantic": "~"}[row["source"]]
    print(mark if row["correct"] is not False else "X", end="", flush=True)


def run_scenario(scenario, workload, threshold):
    """Scenarios A to E."""
    exact, semantic, rows = {}, [], []
    for i, item in enumerate(workload):
        row = new_row(scenario, i, item)
        message = user_message(item)
        key = hashlib.sha256(normalize(message).encode()).hexdigest()
        answer = None

        if scenario in "CDE" and key in exact:
            answer, row["source"], row["correct"] = exact[key], "exact", True

        use_semantic = scenario == "D"
        if answer is None and scenario == "E":
            c = chat(CLASSIFIER_PROMPT, item["question"], 10)
            row["clf_input_tokens"], row["clf_output_tokens"] = c["input_tokens"], c["output_tokens"]
            row["latency_s"] += c["latency_s"]
            row["clf_label"] = "REUSABLE" if "REUTILIZ" in c["text"].upper() else "NOT_REUSABLE"
            use_semantic = row["clf_label"] == "REUSABLE"

        vector = None
        if answer is None and use_semantic:
            e = embed(item["question"])
            vector, row["embedding_tokens"] = e["vector"], e["tokens"]
            row["latency_s"] += e["latency_s"]
            best = max(semantic, key=lambda s: cosine(vector, s["vector"]), default=None)
            if best:
                row["similarity"] = round(cosine(vector, best["vector"]), 4)
                if row["similarity"] >= threshold:
                    answer, row["source"] = best["answer"], "semantic"
                    row["served_answer_key"] = best["answer_key"]
                    row["correct"] = best["answer_key"] == item["answer_key"]

        if answer is None:
            system = f"Sessao {uuid.uuid4()}\n\n{SYSTEM_PROMPT}" if scenario == "A" else SYSTEM_PROMPT
            r = chat(system, message)
            answer = r["text"]
            record_model_call(row, r)
            if scenario in "CDE":
                exact[key] = answer
            if vector is not None:
                semantic.append({"vector": vector, "answer_key": item["answer_key"], "answer": answer})

        finish_row(row, answer, rows)
    print()
    return rows


def numbers_in(text):
    return set(re.findall(r"\d+", re.sub(r",00\b", "", text)))


def run_status_scenario(workload, threshold, k, audit_rate, stamp, digit_rule=False):
    """Scenarios F and G: semantic cache gated by evidence, kept in SQLite.
    Each group of similar questions starts as OBSERVING. When k different users receive answers that
    match the stored one, the group becomes APPROVED and starts answering from the cache.
    One diverging answer makes it BLOCKED, and it is never reused.
    With digit_rule (scenario G), a question that contains a digit skips the semantic cache."""
    scenario = "G" if digit_rule else "F"
    RESULTS.mkdir(exist_ok=True)
    db = sqlite3.connect(RESULTS / f"{stamp}-cache_{scenario.lower()}.sqlite")
    db.execute("create table groups (id integer primary key, status text, sample_question text, answer text, "
               "answer_key text, confirmations integer, users text, served integer default 0)")
    vectors, exact, rows, rnd = [], {}, [], random.Random(99)

    def answer_vector(text, row):
        is_new = text not in EMBEDDINGS
        e = embed(text)
        if is_new:
            row["embedding_tokens"] += e["tokens"]
            row["latency_s"] += e["latency_s"]
        return e["vector"]

    def answers_match(a, b, row):
        return numbers_in(a) == numbers_in(b) and cosine(answer_vector(a, row), answer_vector(b, row)) >= ANSWER_MATCH_THRESHOLD

    for i, item in enumerate(workload):
        row = new_row(scenario, i, item)
        message = user_message(item)
        key = hashlib.sha256(normalize(message).encode()).hexdigest()

        def call_model():
            r = chat(SYSTEM_PROMPT, message)
            record_model_call(row, r)
            exact[key] = r["text"]
            return r["text"]

        if key in exact:
            answer, row["source"], row["correct"] = exact[key], "exact", True
        elif digit_rule and re.search(r"\d", item["question"]):
            answer, row["group_status"] = call_model(), "RULE"
        else:
            e = embed(item["question"])
            vector = e["vector"]
            row["embedding_tokens"], row["latency_s"] = e["tokens"], e["latency_s"]
            best = max(vectors, key=lambda v: cosine(vector, v[0]), default=None)
            gid = best[1] if best and cosine(vector, best[0]) >= threshold else None
            if gid is None:
                answer = call_model()
                gid = db.execute("insert into groups (status, sample_question, answer, answer_key, confirmations, users) "
                                 "values ('OBSERVING', ?, ?, ?, 0, ?)",
                                 (item["question"], answer, item["answer_key"], item["user"])).lastrowid
                row["group_status"] = "NEW"
            else:
                row["similarity"] = round(cosine(vector, best[0]), 4)
                status, stored, stored_key, confirmations, users = db.execute(
                    "select status, answer, answer_key, confirmations, users from groups where id = ?", (gid,)).fetchone()
                row["group_status"] = status
                if status == "APPROVED" and rnd.random() >= audit_rate:
                    answer, row["source"] = stored, "semantic"
                    row["served_answer_key"] = stored_key
                    row["correct"] = stored_key == item["answer_key"]
                    db.execute("update groups set served = served + 1 where id = ?", (gid,))
                else:
                    answer = call_model()
                    if status == "APPROVED":
                        row["audit"] = True
                        if not answers_match(stored, answer, row):
                            db.execute("update groups set status = 'BLOCKED' where id = ?", (gid,))
                    elif status == "OBSERVING":
                        if not answers_match(stored, answer, row):
                            db.execute("update groups set status = 'BLOCKED' where id = ?", (gid,))
                        elif item["user"] not in users.split(","):
                            confirmations += 1
                            db.execute("update groups set confirmations = ?, users = ?, status = ? where id = ?",
                                       (confirmations, users + "," + item["user"],
                                        "APPROVED" if confirmations >= k else "OBSERVING", gid))
            vectors.append((vector, gid))

        finish_row(row, answer, rows)
    print()
    db.commit()
    groups = dict(db.execute("select status, count(*) from groups group by status").fetchall())
    db.close()
    return rows, groups


def summarize(rows):
    n = len(rows)
    cost = sum(r["cost_usd"] for r in rows)
    input_tokens = sum(r["input_tokens"] for r in rows)
    cached = sum(r["cached_tokens"] for r in rows)
    classified = [r for r in rows if r["clf_label"]]
    return {
        "requests": n,
        "model_calls": sum(r["source"] == "model" for r in rows),
        "exact_hits": sum(r["source"] == "exact" for r in rows),
        "semantic_hits": sum(r["source"] == "semantic" for r in rows),
        "wrong_hits": sum(r["correct"] is False for r in rows),
        "input_tokens": input_tokens,
        "cached_tokens": cached,
        "cached_input_pct": round(100 * cached / input_tokens, 1) if input_tokens else 0.0,
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "cost_usd": round(cost, 6),
        "cost_per_1000_usd": round(cost / n * 1000, 4),
        "median_latency_s": round(statistics.median(r["latency_s"] for r in rows), 3),
        "classifier_accuracy": (f"{sum(r['clf_label'] == EXPECTED_LABEL[r['type']] for r in classified)}/{len(classified)}"
                                if classified else "-"),
    }


# ---------- commands ----------

def cmd_simulate(a):
    workload = build_workload(a.n, a.faq_share, a.personal_share, a.seed)
    prompt_tokens = len(SYSTEM_PROMPT) / 3.2 + 40  # rough estimate
    worst = {"A": (a.n, 0, 0), "B": (a.n, 0, 0), "C": (a.n, 0, 0), "D": (a.n, a.n, 0), "E": (a.n, a.n, a.n),
             "F": (a.n, 2 * a.n, 0), "G": (a.n, 2 * a.n, 0)}
    print(f"Workload: {a.n} requests: " + ", ".join(
        f"{sum(w['type'] == t for w in workload)} {t}" for t in ("FAQ", "UNIQUE", "PERSONAL")))
    print(f"System prompt: about {prompt_tokens:.0f} tokens (estimated)\n")
    total_calls, total_usd = 0, 0.0
    for scenario in a.scenarios:
        chat_calls, embedding_calls, classifier_calls = worst[scenario]
        usd = (chat_calls * chat_cost(prompt_tokens, 0, MAX_OUTPUT_TOKENS) + classifier_calls * chat_cost(120, 0, 10)
               + embedding_calls * embedding_cost(20))
        total_calls += chat_calls + embedding_calls + classifier_calls
        total_usd += usd
        print(f"  {scenario}: up to {chat_calls + embedding_calls + classifier_calls:4d} calls, up to US$ {usd:.4f}")
    print(f"\nWorst case (no cache hit at all): {total_calls} calls, US$ {total_usd:.4f}")
    return total_calls


def cmd_run(a):
    total = cmd_simulate(a)
    if total > a.max_calls:
        sys.exit(f"\nThe worst case exceeds the limit of {a.max_calls} calls. Lower --n or raise --max-calls.")
    CallBudget.maximum = a.max_calls
    workload = build_workload(a.n, a.faq_share, a.personal_share, a.seed)
    RESULTS.mkdir(exist_ok=True)
    stamp = (datetime.now().strftime("%Y%m%d-%H%M%S")
             + f"-n{a.n}-faq{int(a.faq_share * 100)}-{a.scenarios}-thr{int(round(a.threshold * 100))}")
    all_rows, summary = [], {}
    print("\nLegend: . model   = exact cache   ~ semantic cache   X wrong answer from cache\n")
    for scenario in a.scenarios:
        print(f"  {scenario} ", end="")
        if scenario in "FG":
            rows, groups = run_status_scenario(workload, a.threshold, a.k, a.audit_rate, stamp, scenario == "G")
        else:
            rows, groups = run_scenario(scenario, workload, a.threshold), None
        all_rows += rows
        summary[scenario] = summarize(rows)
        if groups is not None:
            summary[scenario]["groups"] = groups
            summary[scenario]["audits"] = sum(r["audit"] for r in rows)
    save_embeddings()
    out = {"timestamp": stamp, "params": vars(a) | {"model": CHAT_MODEL, "embeddings": EMBEDDINGS_MODEL},
           "prices": PRICES, "api_calls": CallBudget.made, "summary": summary}
    (RESULTS / f"{stamp}-summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    with (RESULTS / f"{stamp}-rows.jsonl").open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{'sc':3} {'model':>6} {'exact':>5} {'sem':>5} {'wrong':>5} {'%cached':>7} {'US$/1000':>9} {'p50 s':>7} {'clf':>8}")
    for scenario, s in summary.items():
        print(f"{scenario:3} {s['model_calls']:6d} {s['exact_hits']:5d} {s['semantic_hits']:5d} {s['wrong_hits']:5d} "
              f"{s['cached_input_pct']:7.1f} {s['cost_per_1000_usd']:9.4f} {s['median_latency_s']:7.3f} {s['classifier_accuracy']:>8}")
    print(f"\nAPI calls: {CallBudget.made}. Files: {RESULTS / (stamp + '-summary.json')}")


def cmd_calibrate(a):
    CallBudget.maximum = a.max_calls
    items = [(g["group"], p) for block in ("faq", "personal") for g in DATA[block] for p in g["phrasings"]]
    vectors = [embed(p)["vector"] for _, p in items]
    save_embeddings()
    same, different = [], []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            s = cosine(vectors[i], vectors[j])
            (same if items[i][0] == items[j][0] else different).append((s, items[i][1], items[j][1]))
    print(f"{len(items)} phrasings, {CallBudget.made} new embedding calls")
    print(f"Same question:       median {statistics.median(s for s, *_ in same):.3f}, min {min(same)[0]:.3f}")
    print(f"Different questions: median {statistics.median(s for s, *_ in different):.3f}, max {max(different)[0]:.3f}\n")
    print("threshold  paraphrase pairs accepted  different-question pairs accepted")
    for t in [x / 100 for x in range(50, 96, 5)]:
        print(f"  {t:.2f}     {100 * sum(s >= t for s, *_ in same) / len(same):6.1f}%"
              f"                    {100 * sum(s >= t for s, *_ in different) / len(different):6.2f}%")
    print("\nMost similar pairs of different questions:")
    for s, p1, p2 in sorted(different, reverse=True)[:8]:
        print(f"  {s:.3f}  {p1}  |  {p2}")


def cmd_project(a):
    """Project a.n requests for A, B and C without calling the API. In A and B every request goes to
    the model. In C the exact-cache hits are simulated over the workload (they are deterministic).
    Each model call costs the average measured in the real run of the same workload."""
    def mean_call_cost(path, scenario):
        v = [chat_cost(r["input_tokens"], r["cached_tokens"], r["output_tokens"])
             for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
             if r["scenario"] == scenario and r["source"] == "model"]
        return sum(v) / len(v)

    projection = {}
    for faq in (10, 50, 90):
        rows_100 = sorted(RESULTS.glob(f"*-n100-faq{faq}-ABCDE-thr80-rows.jsonl"))[-1]
        rows_300 = sorted(RESULTS.glob(f"*-n300-faq{faq}-C-thr80-rows.jsonl"))[-1]
        cost = {"A": mean_call_cost(rows_100, "A"), "B": mean_call_cost(rows_100, "B"), "C": mean_call_cost(rows_300, "C")}
        workload = build_workload(a.n, faq / 100, a.personal_share, a.seed)
        seen, calls = set(), 0
        for item in workload:
            key = normalize(user_message(item))
            calls += key not in seen
            seen.add(key)
        projection[str(faq)] = {"n": a.n, "A": round(a.n * cost["A"], 4), "B": round(a.n * cost["B"], 4),
                                "C": round(calls * cost["C"], 4), "C_model_calls": calls, "C_exact_hits": a.n - calls}
        print(f"FAQ {faq}%: A US$ {projection[str(faq)]['A']:.4f}  B US$ {projection[str(faq)]['B']:.4f}  "
              f"C US$ {projection[str(faq)]['C']:.4f} ({a.n - calls} exact hits)")
    (RESULTS / "projection.json").write_text(json.dumps(projection, indent=2), encoding="utf-8")


def main():
    global EMBEDDINGS
    p = argparse.ArgumentParser(description="LLM cache layers benchmark")
    p.add_argument("command", choices=["calibrate", "simulate", "run", "project"])
    p.add_argument("--n", type=int, default=30, help="requests in the workload")
    p.add_argument("--faq-share", type=float, default=0.5, help="share of frequent questions among non-personal requests")
    p.add_argument("--personal-share", type=float, default=0.2, help="share of requests that depend on customer data")
    p.add_argument("--threshold", type=float, default=0.75, help="minimum cosine similarity for the semantic cache")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--scenarios", default="ABCDE", help="letters of the scenarios to run")
    p.add_argument("--max-calls", type=int, default=250, help="hard limit of API calls in the run")
    p.add_argument("--k", type=int, default=2, help="confirmations from different users to approve a group (F, G)")
    p.add_argument("--audit-rate", type=float, default=0.1, help="share of approved-group hits that still call the model (F, G)")
    a = p.parse_args()
    EMBEDDINGS = load_embeddings()
    {"calibrate": cmd_calibrate, "simulate": cmd_simulate, "run": cmd_run, "project": cmd_project}[a.command](a)


if __name__ == "__main__":
    main()
