"""Analysis of the stored results: writes the charts to docs/img and prints the tables as Markdown.
Makes no API calls. Requires matplotlib."""
import glob
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lab

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
IMG = ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)
FAQS = (10, 50, 90)
COLOR = {"A": "#7f7f7f", "B": "#1f77b4", "C": "#2ca02c", "D": "#d62728", "E": "#ff7f0e", "F": "#9467bd", "G": "#17becf"}
NAME = {"A": "A no cache", "B": "B prompt", "C": "C prompt+exact", "D": "D naive semantic",
        "E": "E LLM classifier", "F": "F status store", "G": "G status+digit rule"}
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def load(faq, scenario, n=None, threshold=80):
    """Rows of one scenario. A, B and D default to the 100-request runs, the others to 300."""
    n = n or (100 if scenario in "ABD" else 300)
    tag = "ABCDE" if n == 100 else scenario
    path = sorted(glob.glob(str(RESULTS / f"*-n{n}-faq{faq}-{tag}-thr{threshold}-rows.jsonl")))[-1]
    return [r for r in map(json.loads, open(path, encoding="utf-8")) if r["scenario"] == scenario]


def load_summary(faq, scenario, n, threshold=80):
    path = sorted(glob.glob(str(RESULTS / f"*-n{n}-faq{faq}-{scenario}-thr{threshold}-summary.json")))[-1]
    return json.load(open(path, encoding="utf-8"))["summary"][scenario]


def pct(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))] if v else float("nan")


def cost_per_1000(rows):
    return sum(r["cost_usd"] for r in rows) / len(rows) * 1000


def save(name):
    plt.tight_layout()
    plt.savefig(IMG / name)
    plt.close()


D = {(f, s): load(f, s) for f in FAQS for s in "ABCDEFG"}
D300 = [load(f, "D", n=300) for f in FAQS]
model_call_cost = lambda r: lab.chat_cost(r["input_tokens"], r["cached_tokens"], r["output_tokens"])

# ---------- T1: tokens and cost per model call ----------
print("\n## T1 tokens and cost per model call (mean of the three workloads)\n")
print("| Scenario | Calls | Input | Cached | Cached % | Output | US$ per call |")
print("|---|---|---|---|---|---|---|")
for s in "ABCEFG":
    rows = [r for f in FAQS for r in D[f, s] if r["source"] == "model"]
    i, c, o = (statistics.mean(r[x] for r in rows) for x in ("input_tokens", "cached_tokens", "output_tokens"))
    print(f"| {s} | {len(rows)} | {i:.0f} | {c:.0f} | {100 * c / i:.1f} | {o:.0f} | {statistics.mean(map(model_call_cost, rows)):.6f} |")

# ---------- T2: full results ----------
print("\n## T2 results per scenario and workload\n")
print("| FAQ share | Scenario | n | Model | Exact | Semantic | Wrong | US$/1000 | p50 s | p95 s |")
print("|---|---|---|---|---|---|---|---|---|---|")
for f in FAQS:
    for s in "ABCDEFG":
        rows = D[f, s]
        src = Counter(r["source"] for r in rows)
        lat = [r["latency_s"] for r in rows]
        print(f"| {f}% | {s} | {len(rows)} | {src['model']} | {src['exact']} | {src['semantic']} | "
              f"{sum(r['correct'] is False for r in rows)} | {cost_per_1000(rows):.4f} | {pct(lat, 50):.2f} | {pct(lat, 95):.2f} |")

print("\n## T2b scenario D with 300 requests\n")
print("| FAQ share | Model | Exact | Semantic | Wrong | Another customer's data | US$/1000 |")
print("|---|---|---|---|---|---|---|")
for f, rows in zip(FAQS, D300):
    src = Counter(r["source"] for r in rows)
    wrong = [r for r in rows if r["correct"] is False]
    print(f"| {f}% | {src['model']} | {src['exact']} | {src['semantic']} | {len(wrong)} | "
          f"{sum(r['type'] == 'PERSONAL' for r in wrong)} | {cost_per_1000(rows):.4f} |")

# ---------- T3: latency of requests that reached the model ----------
print("\n## T3 latency of requests that reached the model (s)\n")
print("| Scenario | n | p50 | p90 | p99 | mean |")
print("|---|---|---|---|---|---|")
for s in "ABCEFG":
    v = [r["latency_s"] for f in FAQS for r in D[f, s] if r["source"] == "model"]
    print(f"| {s} | {len(v)} | {pct(v, 50):.2f} | {pct(v, 90):.2f} | {pct(v, 99):.2f} | {statistics.mean(v):.2f} |")

# ---------- T4: classifier ----------
print("\n## T4 classifier confusion matrix (E, 3 x 300 requests)\n")
m = Counter((r["type"], r["clf_label"]) for f in FAQS for r in D[f, "E"] if r["clf_label"])
print("| True type | Expected | REUSABLE | NOT_REUSABLE | Accuracy |")
print("|---|---|---|---|---|")
for t in ("FAQ", "UNIQUE", "PERSONAL"):
    yes, no = m[t, "REUSABLE"], m[t, "NOT_REUSABLE"]
    print(f"| {t} | {lab.EXPECTED_LABEL[t]} | {yes} | {no} | {100 * (yes if t == 'FAQ' else no) / (yes + no):.1f}% |")
clf = [r for f in FAQS for r in D[f, "E"] if r["clf_label"]]
print(f"\nclassifier: mean input {statistics.mean(r['clf_input_tokens'] for r in clf):.0f} tokens, "
      f"output {statistics.mean(r['clf_output_tokens'] for r in clf):.1f}, mean cost US$ "
      f"{statistics.mean(lab.chat_cost(r['clf_input_tokens'], 0, r['clf_output_tokens']) for r in clf):.7f}")

# ---------- T5: semantic hits by question type ----------
print("\n## T5 semantic hits by question type\n")
print("| Scenario | Type | Right | Wrong | Precision |")
print("|---|---|---|---|---|")
for s in "DEFG":
    for t in ("FAQ", "UNIQUE", "PERSONAL"):
        v = [r["correct"] for f in FAQS for r in D[f, s] if r["type"] == t and r["source"] == "semantic"]
        if v:
            print(f"| {s} | {t} | {sum(v)} | {len(v) - sum(v)} | {100 * sum(v) / len(v):.0f}% |")

# ---------- T6: group status ----------
print("\n## T6 group status at the end (F and G, 300 requests)\n")
print("| Scenario | FAQ share | Approved | Observing | Blocked | Answers served |")
print("|---|---|---|---|---|---|")
for s in "FG":
    for f in FAQS:
        r = load_summary(f, s, 300)
        g = r["groups"]
        print(f"| {s} | {f}% | {g.get('APPROVED', 0)} | {g.get('OBSERVING', 0)} | {g.get('BLOCKED', 0)} | {r['semantic_hits']} |")
print("\nF: requests by type and status of the matched group:",
      dict(Counter((r["type"], r["group_status"]) for f in FAQS for r in D[f, "F"] if r["group_status"])))

# ---------- fig 1: measured cost ----------
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for a, f in zip(ax, FAQS):
    v = [cost_per_1000(D[f, s]) for s in "ABCDEFG"]
    bars = a.bar(list("ABCDEFG"), v, color=[COLOR[s] for s in "ABCDEFG"])
    a.bar_label(bars, fmt="%.3f", fontsize=7)
    a.set_title(f"{f}% FAQ")
ax[0].set_ylabel("US$ per 1,000 requests (measured)")
fig.suptitle("Measured cost per scenario (A, B, D: n=100; C, E, F, G: n=300)", fontsize=10)
save("fig1_measured_cost.png")

# ---------- fig 2: cost breakdown ----------
fig, ax = plt.subplots(figsize=(8, 3.6))
base = [0.0] * 7
for part, color in zip(["uncached input", "cached input", "output", "classifier", "embeddings"],
                       ["#4c72b0", "#a6c8e8", "#dd8452", "#c44e52", "#8172b3"]):
    v = [sum(lab.row_cost_parts(r)[part] for r in D[50, s]) / len(D[50, s]) * 1000 for s in "ABCDEFG"]
    ax.bar([NAME[s] for s in "ABCDEFG"], v, bottom=base, label=part, color=color)
    base = [x + y for x, y in zip(base, v)]
ax.set_ylabel("US$ per 1,000 requests")
ax.set_title("Cost breakdown, 50% FAQ workload")
ax.legend(fontsize=7)
plt.xticks(rotation=20, ha="right")
save("fig2_cost_breakdown.png")

# ---------- fig 3: prompt cache per call ----------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
for f in FAQS:
    ax[0].plot([100 * r["cached_tokens"] / r["input_tokens"] for r in D[f, "B"]], lw=0.9, label=f"{f}% FAQ")
ax[0].set_xlabel("request index (scenario B)")
ax[0].set_ylabel("% of input tokens read from cache")
ax[0].legend(fontsize=7)
shares = [100 * r["cached_tokens"] / r["input_tokens"] for f in FAQS for r in D[f, "B"]]
ax[1].hist(shares, bins=20, color=COLOR["B"])
ax[1].set_xlabel("% of input tokens read from cache")
ax[1].set_ylabel("calls")
zeros = sum(x == 0 for x in shares)
fig.suptitle(f"Prompt cache per call (B, 300 calls, {zeros} with no cached token)", fontsize=10)
save("fig3_prompt_cache.png")
print(f"\nB: calls with no cached token {zeros}/{len(shares)}; median cached share {statistics.median(shares):.1f}%")

# ---------- fig 4: latency ----------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for s in "AB":
    v = sorted(r["latency_s"] for f in FAQS for r in D[f, s])
    ax[0].plot(v, [i / len(v) for i in range(len(v))], label=NAME[s], color=COLOR[s])
ax[0].set_xlabel("model call latency (s)")
ax[0].set_ylabel("cumulative fraction")
ax[0].set_title("Model call, with and without prompt cache")
for s in "CEFG":
    v = sorted(r["latency_s"] for r in D[50, s])
    ax[1].plot(v, [i / len(v) for i in range(len(v))], label=NAME[s], color=COLOR[s])
ax[1].set_xlabel("latency per request (s)")
ax[1].set_title("End to end, 50% FAQ workload")
for a in ax:
    a.set_xlim(0, 6)
    a.legend(fontsize=7)
save("fig4_latency.png")

# ---------- fig 5 and 6: similarity and threshold ----------
emb = json.load(open(RESULTS / "embeddings.json", encoding="utf-8"))
items = [(g["group"], p) for block in ("faq", "personal") for g in lab.DATA[block] for p in g["phrasings"]]
same, different = [], []
for i in range(len(items)):
    for j in range(i + 1, len(items)):
        s = lab.cosine(emb[items[i][1]]["vector"], emb[items[j][1]]["vector"])
        (same if items[i][0] == items[j][0] else different).append(s)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
bins = [x / 50 for x in range(0, 51)]
ax[0].hist(different, bins=bins, alpha=0.6, density=True, label=f"different questions (n={len(different)})", color="#d62728")
ax[0].hist(same, bins=bins, alpha=0.6, density=True, label=f"same question (n={len(same)})", color="#1f77b4")
ax[0].set_xlabel("cosine similarity between two phrasings")
ax[0].set_ylabel("density")
ts = [x / 100 for x in range(40, 100)]
ax[1].plot(ts, [100 * sum(s >= t for s in same) / len(same) for t in ts], label="paraphrase pairs accepted (recall)", color="#1f77b4")
ax[1].plot(ts, [100 * sum(s >= t for s in different) / len(different) for t in ts], label="different-question pairs accepted", color="#d62728")
ax[1].set_xlabel("threshold")
ax[1].set_ylabel("% of pairs")
for a in ax:
    a.axvline(0.80, color="k", ls="--", lw=0.8)
    a.legend(fontsize=7)
fig.suptitle("Similarity between the 70 fixed phrasings (text-embedding-3-small)", fontsize=10)
save("fig5_similarity.png")
print("\n## T7 threshold vs accepted pairs\n\n| Threshold | Paraphrase pairs accepted | Different-question pairs accepted | Precision among accepted pairs |\n|---|---|---|---|")
for t in (0.60, 0.70, 0.75, 0.80, 0.85, 0.90):
    a, b = sum(s >= t for s in same), sum(s >= t for s in different)
    print(f"| {t:.2f} | {100 * a / len(same):.1f}% | {100 * b / len(different):.2f}% | {100 * a / (a + b):.0f}% |")
print(f"\nsame: median {statistics.median(same):.3f} min {min(same):.3f}; different: median {statistics.median(different):.3f} max {max(different):.3f}")

fig, ax = plt.subplots(figsize=(7, 3.4))
rows_d = [r for f in FAQS for r in D[f, "D"] if r["similarity"] is not None]
buckets = {"right answer from cache": [r["similarity"] for r in rows_d if r["correct"] is True],
           "wrong answer from cache": [r["similarity"] for r in rows_d if r["correct"] is False],
           "below threshold": [r["similarity"] for r in rows_d if r["source"] != "semantic"]}
ax.hist(list(buckets.values()), bins=[x / 40 for x in range(8, 41)], stacked=True,
        label=[f"{k} (n={len(v)})" for k, v in buckets.items()], color=["#1f77b4", "#d62728", "#bbbbbb"])
ax.axvline(0.80, color="k", ls="--", lw=0.8)
ax.set_xlabel("similarity to the nearest cache entry (scenario D)")
ax.set_ylabel("requests")
ax.legend(fontsize=7)
save("fig6_similarity_scenario_d.png")
wrong = buckets["wrong answer from cache"]
print(f"\nD: wrong hits with similarity >= 0.95: {sum(s >= 0.95 for s in wrong)}/{len(wrong)}; "
      f"median similarity wrong {statistics.median(wrong):.3f}, right {statistics.median(buckets['right answer from cache']):.3f}")

# ---------- fig 7: volume ----------
call_cost = {s: statistics.mean(model_call_cost(r) for f in FAQS for r in D[f, s] if r["source"] == "model") for s in "ABC"}
sizes = [50, 100, 200, 300, 500, 1000, 2000, 5000]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
print("\n## T8 exact cache vs volume (simulated, seed 11)\n\n| FAQ share | Metric | " + " | ".join(str(n) for n in sizes) + " |\n|---|---|" + "---|" * len(sizes))
for f, ls in zip(FAQS, (":", "--", "-")):
    seen, hits, rate, cost = set(), 0, [], []
    for i, item in enumerate(lab.build_workload(max(sizes), f / 100, 0.2, 11), 1):
        key = lab.normalize(lab.user_message(item))
        hits += key in seen
        seen.add(key)
        if i in sizes:
            rate.append(100 * hits / i)
            cost.append((i - hits) * call_cost["C"] / i * 1000)
    ax[0].plot(sizes, rate, ls, marker="o", ms=3, color=COLOR["C"], label=f"{f}% FAQ")
    ax[1].plot(sizes, cost, ls, marker="o", ms=3, color=COLOR["C"], label=f"C, {f}% FAQ")
    print(f"| {f}% | hit rate | " + " | ".join(f"{t:.0f}%" for t in rate) + " |")
    print(f"| {f}% | US$/1000 | " + " | ".join(f"{c:.4f}" for c in cost) + " |")
ax[1].axhline(call_cost["A"] * 1000, color=COLOR["A"], lw=1, label="A")
ax[1].axhline(call_cost["B"] * 1000, color=COLOR["B"], lw=1, label="B")
for a in ax:
    a.set_xscale("log")
    a.set_xlabel("cumulative requests")
    a.legend(fontsize=7)
ax[0].set_ylabel("% answered by the exact cache")
ax[1].set_ylabel("US$ per 1,000 requests")
fig.suptitle("Effect of volume on the exact cache (deterministic simulation over the workload)", fontsize=10)
save("fig7_volume.png")
print(f"\nmean cost per model call: A {call_cost['A']:.6f} B {call_cost['B']:.6f} C {call_cost['C']:.6f}")

# ---------- fig 8: cumulative cost ----------
fig, ax = plt.subplots(figsize=(7.5, 3.6))
x = list(range(1, 301))
for s in "AB":
    ax.plot(x, [i * call_cost[s] for i in x], color=COLOR[s], ls="--", label=f"{NAME[s]} (mean per call)")
for s in "CEFG":
    total, curve = 0.0, []
    for r in D[50, s]:
        total += r["cost_usd"]
        curve.append(total)
    ax.plot(x, curve, color=COLOR[s], label=NAME[s])
ax.set_xlabel("requests")
ax.set_ylabel("cumulative US$")
ax.set_title("Cumulative cost, 50% FAQ workload")
ax.legend(fontsize=7)
save("fig8_cumulative_cost.png")

# ---------- fig 0: the four cache types by name (README chart) ----------
def wrong_from_cache(groups):
    served = [r for rows in groups for r in rows if r["source"] != "model"]
    return sum(r["correct"] is False for r in served), len(served)


types = [
    ("No cache", [D[f, "A"] for f in FAQS]),
    ("Prompt cache", [D[f, "B"] for f in FAQS]),
    ("Prompt + exact", [D[f, "C"] for f in FAQS]),
    ("Prompt + exact\n+ semantic", D300),
    ("Prompt + exact\n+ semantic with\nstatus store", [D[f, "F"] for f in FAQS]),
]
names = [t[0] for t in types][::-1]
costs = [statistics.mean(cost_per_1000(rows) for rows in t[1]) for t in types][::-1]
wrongs = [wrong_from_cache(t[1]) for t in types][::-1]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
bars = ax[0].barh(names, costs, color="#1f77b4")
ax[0].bar_label(bars, labels=[f"US$ {c:.3f}" for c in costs], padding=3, fontsize=8)
ax[0].set_xlim(0, max(costs) * 1.22)
ax[0].set_xlabel("US$ per 1,000 requests")
ax[0].set_title("Cost", fontsize=10)
share = [100 * w / n if n else 0 for w, n in wrongs]
bars = ax[1].barh(names, share, color="#d62728")
ax[1].bar_label(bars, labels=[f"{p:.0f}%  ({w} of {n})" if n else "no answers from cache" for p, (w, n) in zip(share, wrongs)],
                padding=3, fontsize=8)
ax[1].set_xlim(0, 100)
ax[1].set_xlabel("% wrong among the answers served from cache")
ax[1].set_title("Wrong answers", fontsize=10)
for a in ax:
    a.grid(axis="y", visible=False)
fig.suptitle("Mean of the three workloads, 300 requests each (no cache and prompt cache: 100 each)", fontsize=9)
save("fig0_cache_types.png")
print("\n## fig0")
for n, c, (w, t) in zip(names[::-1], costs[::-1], wrongs[::-1]):
    print(f"{n!r}: US$ {c:.4f}/1000 ({100 * (1 - c / costs[-1]):.0f}% lower), wrong {w}/{t}")

# ---------- T9: scenario G with 1,000 requests ----------
print("\n## T9 scenario G with 1,000 requests\n")
print("| FAQ share | Threshold | Calls with exact only | Calls in G | Avoided by semantic | Workload ceiling | Semantic answers | Wrong | US$/1000 |")
print("|---|---|---|---|---|---|---|---|---|")
for f in FAQS:
    workload = lab.build_workload(1000, f / 100, 0.2, 7)
    seen, exact_only, faq_calls = set(), 0, 0
    for item in workload:
        key = lab.normalize(lab.user_message(item))
        if key not in seen:
            exact_only += 1
            faq_calls += item["type"] == "FAQ"
        seen.add(key)
    ceiling = faq_calls - len({w["group"] for w in workload if w["type"] == "FAQ"})
    for thr in (70, 80):
        rows = load(f, "G", n=1000, threshold=thr)
        src = Counter(r["source"] for r in rows)
        print(f"| {f}% | 0.{thr} | {exact_only} | {src['model']} | {exact_only - src['model']} | {ceiling} | {src['semantic']} | "
              f"{sum(r['correct'] is False for r in rows)} | {cost_per_1000(rows):.4f} |")
print("\ncharts:", sorted(p.name for p in IMG.glob("*.png")))
