"""
full_experiment.py — the entire project in one file.

Merges nine separate modules into one script, run top to bottom in four
stages. This exists purely for convenience (one file to upload/manage); the
logic is identical to the split-file version.

STAGE 1 — Main experiment: centralised vs peer-to-peer, 234 runs on real
          GSM8K (Cobbe et al., 2021) and Deal-or-No-Deal (Lewis et al., 2017)
          data. Writes results_raw.csv / results_summary.csv.
STAGE 2 — Sensitivity analysis: does the result hold if our error-rate
          assumption is wrong? Writes sensitivity_error_rate.csv /
          sensitivity_seeds.csv.
STAGE 3 — Fix comparison: two targeted fixes for peer-to-peer's diagnosed
          failure modes, before vs after. Writes fix_comparison_raw.csv /
          fix_comparison_summary.csv.
STAGE 4 — Charts: 6 PNGs, 3 for the main experiment, 3 for the fix results.

Requires real_data/gsm8k_test.jsonl and real_data/dond_test.txt to be
present in the working directory. No live LLM / API key is used anywhere —
every agent's reasoning step is a deterministic simulation policy.

Run with: python full_experiment.py
"""

import csv
import json
import os
import random
import re
import statistics as stats
import time
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


# =====================================================================
# SECTION 1 — AGENT POLICY (originally agent_policy.py)
#
# Each agent's reasoning step is a deterministic simulation policy rather
# than a language model: it applies the agent's assigned rule to the value
# it receives, with a fixed probability of introducing a transmission
# error. Error rates are set, not fitted, to reflect the qualitative
# pattern reported in Liu et al. (2023): BOLAA's controller-based advantage
# shrinks as agents pass along more mis-read information hop by hop. The
# peer-to-peer condition is therefore given a higher per-hop error rate
# than the centralised condition, whose controller re-checks each handoff.
# =====================================================================

CENTRALISED_ERROR_RATE = 0.03   # controller re-validates each handoff
P2P_ERROR_RATE = 0.09           # no controller to catch a mis-read value
CENTRALISED_MISREPORT_RATE = 0.05
P2P_MISREPORT_RATE = 0.05


def apply_noise(true_value, error_rate, rng):
    """Simulate an agent occasionally mis-applying its own rule."""
    if rng.random() < error_rate:
        return true_value + rng.choice([-3, -2, -1, 1, 2, 3])
    return true_value


def agent_step(prev_value, rule, error_rate, rng):
    """One agent's contribution to a decomposition-chain task."""
    mult, add = rule
    correct = prev_value * mult + add
    reported = apply_noise(correct, error_rate, rng)
    message_text = (
        f"Received value {prev_value}. Applying rule (x{mult}{'+' if add>=0 else ''}{add}). "
        f"Passing on {reported}."
    )
    return reported, message_text


def agent_report_valuation(true_vals, misreport_rate, rng):
    """An agent's (possibly distorted) self-report of its own valuations."""
    if rng.random() < misreport_rate:
        idx = rng.randrange(len(true_vals))
        distorted = list(true_vals)
        shift = rng.uniform(-15, 15)
        distorted[idx] = max(0.0, distorted[idx] + shift)
        return distorted
    return list(true_vals)


def accept_trade(item_value_to_holder, holder_avg_item_value):
    """Deterministic stand-in for a negotiating agent's decision on whether
    to give up an item it currently holds. Accepts if the item is worth
    less to the holder than their own average item value."""
    return item_value_to_holder < holder_avg_item_value


# =====================================================================
# SECTION 2 — REAL-DATASET TASK CONSTRUCTION (originally real_tasks.py)
#
# Task Family A (decomposition): built from GSM8K (Cobbe et al., 2021).
# Calculator-annotated solution steps are chained into genuine multi-hop
# tasks wherever one step is an affine function of the previous step's
# result. Task Family B (negotiation): built from the Deal-or-No-Deal
# corpus (Lewis et al., 2017), using real per-agent private valuations.
# =====================================================================

GSM8K_PATH = "real_data/gsm8k_test.jsonl"
DOND_PATH = "real_data/dond_test.txt"

CALC_RE = re.compile(r"<<([^=<>]+)=(-?[\d\.]+)>>")


def _to_num(s):
    s = s.replace(",", "")
    v = float(s)
    return int(v) if v == int(v) else v


def _extract_affine_rule(expr, prev_result):
    expr = expr.strip().replace(" ", "")
    prev_str_variants = {str(prev_result), f"{prev_result:.2f}".rstrip("0").rstrip(".")}
    for op in ["+", "*", "/", "-"]:
        if op not in expr:
            continue
        parts = expr.split(op, 1)
        if len(parts) != 2:
            continue
        left, right = parts
        try:
            left_is_prev = left in prev_str_variants
            right_is_prev = right in prev_str_variants
            if left_is_prev and not right_is_prev:
                c = float(right)
                if op == "+":
                    return (1.0, c)
                if op == "-":
                    return (1.0, -c)
                if op == "*":
                    return (c, 0.0)
                if op == "/":
                    return (1.0 / c, 0.0) if c != 0 else None
            elif right_is_prev and not left_is_prev:
                c = float(left)
                if op == "+":
                    return (1.0, c)
                if op == "-":
                    return (-1.0, c)
                if op == "*":
                    return (c, 0.0)
        except ValueError:
            continue
    return None


def build_gsm8k_chain_tasks(agent_counts=(2, 3, 4, 5), per_bucket=15, seed=32):
    problems = []
    with open(GSM8K_PATH) as f:
        for line in f:
            problems.append(json.loads(line))

    buckets = {k: [] for k in agent_counts}
    for idx, prob in enumerate(problems):
        ans = prob["answer"]
        calc = CALC_RE.findall(ans)
        if len(calc) < min(agent_counts):
            continue
        hops = [(expr, _to_num(res)) for expr, res in calc]
        chain = [hops[0]]
        for expr, res in hops[1:]:
            prev_result = chain[-1][1]
            rule = _extract_affine_rule(expr, prev_result)
            if rule is None:
                break
            chain.append((expr, res, rule))
        chain_len = len(chain)
        if chain_len in buckets and chain[0][1] is not None:
            rules = [h[2] for h in chain[1:]]
            buckets[chain_len].append({
                "problem_idx": idx, "question": prob["question"],
                "seed_value": chain[0][1], "rules": rules, "answer": chain[-1][1],
            })

    rng = random.Random(seed)
    tasks = []
    for k in agent_counts:
        pool = buckets.get(k, [])
        rng.shuffle(pool)
        for item in pool[:per_bucket]:
            tasks.append({
                "id": f"A-{k}-{item['problem_idx']}", "family": "decomposition",
                "agent_count": k, "seed_value": item["seed_value"], "rules": item["rules"],
                "answer": item["answer"], "source_question": item["question"],
                "source": "GSM8K (Cobbe et al., 2021)",
            })
    return tasks


def parse_dond_file(path):
    pool = defaultdict(set)
    tag_re = re.compile(r"<input>\s*(.*?)\s*</input>.*?<partner_input>\s*(.*?)\s*</partner_input>")
    with open(path) as f:
        for line in f:
            m = tag_re.search(line)
            if not m:
                continue
            own = [int(x) for x in m.group(1).split()]
            partner = [int(x) for x in m.group(2).split()]
            own_counts = (own[0], own[2], own[4])
            own_vals = (own[1], own[3], own[5])
            partner_counts = (partner[0], partner[2], partner[4])
            partner_vals = (partner[1], partner[3], partner[5])
            pool[own_counts].add(own_vals)
            pool[partner_counts].add(partner_vals)
    return pool


def build_dond_negotiation_tasks(agent_counts=(2, 3, 4, 5), per_bucket=15, seed=32):
    pool = parse_dond_file(DOND_PATH)
    rng = random.Random(seed)
    tasks = []
    tid = 0
    counts_ranked = sorted(pool.keys(), key=lambda c: -len(pool[c]))

    for k in agent_counts:
        built = 0
        for counts in counts_ranked:
            vecs = list(pool[counts])
            if len(vecs) < k:
                continue
            rng.shuffle(vecs)
            n_groups = len(vecs) // k
            for g in range(n_groups):
                if built >= per_bucket:
                    break
                group = vecs[g * k:(g + 1) * k]
                n_items = len(counts)
                item_counts = counts
                valuations = []
                for agent_vals in group:
                    flat = []
                    for t in range(n_items):
                        flat.extend([agent_vals[t]] * item_counts[t])
                    valuations.append(flat)
                total_items = sum(item_counts)
                if total_items == 0:
                    continue
                optimal_alloc = [max(range(k), key=lambda a: valuations[a][item]) for item in range(total_items)]
                optimal_total_utility = sum(valuations[a][item] for item, a in enumerate(optimal_alloc))
                tasks.append({
                    "id": f"B-{k}-{tid}", "family": "negotiation", "agent_count": k,
                    "n_items": total_items, "valuations": valuations,
                    "optimal_alloc": optimal_alloc, "optimal_total_utility": optimal_total_utility,
                    "source_item_counts": item_counts, "source": "Deal-or-No-Deal corpus (Lewis et al., 2017)",
                })
                tid += 1
                built += 1
            if built >= per_bucket:
                break
    return tasks


# =====================================================================
# SECTION 3 — CENTRALISED COORDINATION (originally centralised.py)
# =====================================================================

def run_centralised_decomposition(task, rng):
    start = time.perf_counter()
    messages = []
    value = task["seed_value"]
    messages.append(f"[Controller] Assigning chain of length {task['agent_count']} to workers. Seed value = {value}.")
    for i, rule in enumerate(task["rules"]):
        value, msg = agent_step(value, rule, CENTRALISED_ERROR_RATE, rng)
        messages.append(f"[Worker {i+1} -> Controller] {msg}")
        messages.append(f"[Controller] Acknowledged worker {i+1}, forwarding value to next worker.")
    messages.append(f"[Controller] Final assembled answer: {value}.")
    elapsed = time.perf_counter() - start
    correct = (abs(value - task["answer"]) < 1e-6)
    return {"correct": correct, "final_value": value, "messages": messages,
            "n_messages": len(messages), "n_turns": task["agent_count"],
            "n_tokens": sum(len(m.split()) for m in messages), "wall_time": elapsed}


def run_centralised_negotiation(task, rng):
    start = time.perf_counter()
    messages = []
    k = task["agent_count"]
    n_items = task["n_items"]
    messages.append(f"[Mediator] Requesting private valuations from {k} agents over {n_items} items.")
    reported = []
    for a in range(k):
        rep = agent_report_valuation(task["valuations"][a], CENTRALISED_MISREPORT_RATE, rng)
        reported.append(rep)
        messages.append(f"[Agent {a+1} -> Mediator] Reported valuation vector.")
    for a in range(k):
        s = sum(reported[a])
        if s > 0 and abs(s - 100) > 20:
            reported[a] = [100 * v / s for v in reported[a]]
            messages.append(f"[Mediator] Renormalised inconsistent report from Agent {a+1}.")
    alloc = [max(range(k), key=lambda a: reported[a][item]) for item in range(n_items)]
    messages.append("[Mediator] Final allocation assigned by mediator based on reported valuations.")
    total_true_utility = sum(task["valuations"][a][item] for item, a in enumerate(alloc))
    elapsed = time.perf_counter() - start
    efficiency = total_true_utility / task["optimal_total_utility"] if task["optimal_total_utility"] > 0 else 0
    return {"correct": efficiency >= 0.90, "efficiency": efficiency, "messages": messages,
            "n_messages": len(messages), "n_turns": k + 1,
            "n_tokens": sum(len(m.split()) for m in messages), "wall_time": elapsed}


# =====================================================================
# SECTION 4 — PEER-TO-PEER COORDINATION, ORIGINAL (originally peer_to_peer.py)
# =====================================================================

MAX_ROUNDS = 6


def run_p2p_decomposition(task, rng):
    start = time.perf_counter()
    messages = []
    value = task["seed_value"]
    messages.append(f"[Agent 1] Starting chain with seed value {value}.")
    for i, rule in enumerate(task["rules"]):
        value, msg = agent_step(value, rule, P2P_ERROR_RATE, rng)
        messages.append(f"[Agent {i+1} -> Agent {i+2 if i+1 < task['agent_count'] else 'final'}] {msg}")
    messages.append(f"[Agent {task['agent_count']}] Reports final value: {value}.")
    elapsed = time.perf_counter() - start
    correct = (abs(value - task["answer"]) < 1e-6)
    return {"correct": correct, "final_value": value, "messages": messages,
            "n_messages": len(messages), "n_turns": task["agent_count"],
            "n_tokens": sum(len(m.split()) for m in messages), "wall_time": elapsed}


def run_p2p_negotiation(task, rng):
    start = time.perf_counter()
    messages = []
    k = task["agent_count"]
    n_items = task["n_items"]
    true_vals = task["valuations"]
    alloc = [i % k for i in range(n_items)]
    messages.append("[Agent 1] Proposing initial allocation (round-robin baseline).")

    rounds = 0
    converged = False
    for r in range(MAX_ROUNDS):
        rounds += 1
        proposer = r % k
        improved = False
        for item in range(n_items):
            holder = alloc[item]
            if holder == proposer:
                continue
            if true_vals[proposer][item] > true_vals[holder][item] * 0.6:
                holder_avg = sum(true_vals[holder]) / n_items
                if accept_trade(true_vals[holder][item], holder_avg):
                    alloc[item] = proposer
                    messages.append(f"[Agent {proposer+1} <-> Agent {holder+1}] Negotiated transfer of item {item}.")
                    improved = True
        if not improved:
            messages.append(f"[Agent {proposer+1}] No further improving trade found this round.")
        if r >= 2 and not improved:
            converged = True
            messages.append("[All agents] No agent proposes further trades; allocation accepted.")
            break

    total_true_utility = sum(true_vals[a][it] for it, a in enumerate(alloc))
    elapsed = time.perf_counter() - start
    efficiency = total_true_utility / task["optimal_total_utility"] if task["optimal_total_utility"] > 0 else 0
    non_convergence = not converged
    return {"correct": efficiency >= 0.90 and not non_convergence, "efficiency": efficiency,
            "non_convergence": non_convergence, "messages": messages, "n_messages": len(messages),
            "n_turns": rounds, "n_tokens": sum(len(m.split()) for m in messages), "wall_time": elapsed}


# =====================================================================
# SECTION 5 — PEER-TO-PEER COORDINATION, FIXED (originally peer_to_peer_fixed.py)
#
# FIX 1 (decomposition): each agent self-checks its own calculation before
# passing it on, by independently recomputing the step and comparing.
# FIX 2 (negotiation): requires a FULL PASS (k consecutive turns) with zero
# successful trades before declaring convergence, instead of stopping after
# just one agent's turn finds nothing - the original's premature-stopping bug.
# =====================================================================

SELF_CHECK_CATCH_RATE = 0.7
MAX_ROUNDS_FIXED = 30


def run_p2p_decomposition_fixed(task, rng):
    start = time.perf_counter()
    messages = []
    value = task["seed_value"]
    messages.append(f"[Agent 1] Starting chain with seed value {value}.")
    for i, rule in enumerate(task["rules"]):
        reported, msg = agent_step(value, rule, P2P_ERROR_RATE, rng)
        mult, add = rule
        correct_value = value * mult + add
        if reported != correct_value and rng.random() < SELF_CHECK_CATCH_RATE:
            messages.append(f"[Agent {i+1}] Self-check caught a transmission error ({reported} -> corrected to {correct_value}).")
            reported = correct_value
        value = reported
        messages.append(f"[Agent {i+1} -> Agent {i+2 if i+1 < task['agent_count'] else 'final'}] {msg}")
    messages.append(f"[Agent {task['agent_count']}] Reports final value: {value}.")
    elapsed = time.perf_counter() - start
    correct = (abs(value - task["answer"]) < 1e-6)
    return {"correct": correct, "final_value": value, "messages": messages,
            "n_messages": len(messages), "n_turns": task["agent_count"],
            "n_tokens": sum(len(m.split()) for m in messages), "wall_time": elapsed}


def run_p2p_negotiation_fixed(task, rng):
    start = time.perf_counter()
    messages = []
    k = task["agent_count"]
    n_items = task["n_items"]
    true_vals = task["valuations"]
    alloc = [i % k for i in range(n_items)]
    messages.append("[Agent 1] Proposing initial allocation (round-robin baseline).")

    rounds = 0
    consecutive_no_improvement = 0
    converged = False
    for r in range(MAX_ROUNDS_FIXED):
        rounds += 1
        proposer = r % k
        improved = False
        for item in range(n_items):
            holder = alloc[item]
            if holder == proposer:
                continue
            if true_vals[proposer][item] > true_vals[holder][item] * 0.6:
                holder_avg = sum(true_vals[holder]) / n_items
                if accept_trade(true_vals[holder][item], holder_avg):
                    alloc[item] = proposer
                    messages.append(f"[Agent {proposer+1} <-> Agent {holder+1}] Negotiated transfer of item {item}.")
                    improved = True

        if improved:
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1
            messages.append(f"[Agent {proposer+1}] No further improving trade found this turn.")

        if consecutive_no_improvement >= k:
            converged = True
            messages.append("[All agents] Full pass with no improving trades found; allocation accepted.")
            break

    total_true_utility = sum(true_vals[a][it] for it, a in enumerate(alloc))
    elapsed = time.perf_counter() - start
    efficiency = total_true_utility / task["optimal_total_utility"] if task["optimal_total_utility"] > 0 else 0
    non_convergence = not converged
    return {"correct": efficiency >= 0.90 and not non_convergence, "efficiency": efficiency,
            "non_convergence": non_convergence, "messages": messages, "n_messages": len(messages),
            "n_turns": rounds, "n_tokens": sum(len(m.split()) for m in messages), "wall_time": elapsed}


# =====================================================================
# SECTION 6 — STAGE 1: MAIN EXPERIMENT (originally run_experiment.py)
# =====================================================================

MASTER_SEED = 32
TASKS_PER_BUCKET = 15


def tag_failure_decomposition(result, task):
    if result["correct"]:
        return None
    diff = result["final_value"] - task["answer"]
    if abs(diff) <= 3:
        return "Information Loss (single mis-transmitted value)"
    return "Compounding Information Loss (multiple mis-transmitted values)"


def tag_failure_negotiation(result):
    if result["correct"]:
        return None
    if result.get("non_convergence"):
        return "Non-Convergence (no agreement reached within round limit)"
    return "Inefficient Allocation (agreement reached, but below efficiency threshold)"


def run_main_experiment():
    rng = random.Random(MASTER_SEED)
    decomposition_tasks = build_gsm8k_chain_tasks(per_bucket=TASKS_PER_BUCKET, seed=MASTER_SEED)
    negotiation_tasks = build_dond_negotiation_tasks(per_bucket=TASKS_PER_BUCKET, seed=MASTER_SEED)

    fieldnames = ["task_id", "family", "condition", "agent_count", "correct",
                  "n_messages", "n_turns", "n_tokens", "wall_time", "failure_tag", "efficiency"]

    done_keys = set()
    rows = []
    if os.path.exists("results_raw.csv"):
        with open("results_raw.csv", newline="") as f:
            for r in csv.DictReader(f):
                r["correct"] = r["correct"] in ("True", "1", "true")
                r["agent_count"] = int(r["agent_count"])
                r["n_messages"] = float(r["n_messages"])
                r["n_turns"] = float(r["n_turns"])
                r["n_tokens"] = float(r["n_tokens"])
                r["wall_time"] = float(r["wall_time"])
                rows.append(r)
                done_keys.add((r["task_id"], r["condition"]))
        if done_keys:
            print(f"Resuming: found {len(done_keys)} already-completed runs in results_raw.csv, skipping those.")

    write_header = not os.path.exists("results_raw.csv")
    csv_file = open("results_raw.csv", "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    def save_row(row):
        row.setdefault("efficiency", "")
        writer.writerow(row)
        csv_file.flush()
        rows.append(row)

    def safe_run(run_fn, task, label):
        try:
            return run_fn(task, rng)
        except Exception as e:
            print(f"  [FAILED: {label} on {task['id']} - {str(e)[:120]} - logging as RUN_ERROR and continuing]")
            return {"correct": False, "n_messages": 0, "n_turns": 0, "n_tokens": 0,
                    "wall_time": 0.0, "efficiency": 0.0, "_run_error": True}

    for task in decomposition_tasks:
        if (task["id"], "centralised") not in done_keys:
            r_c = safe_run(run_centralised_decomposition, task, "centralised decomposition")
            save_row({"task_id": task["id"], "family": "decomposition", "condition": "centralised",
                      "agent_count": task["agent_count"], "correct": r_c["correct"],
                      "n_messages": r_c["n_messages"], "n_turns": r_c["n_turns"],
                      "n_tokens": r_c["n_tokens"], "wall_time": r_c["wall_time"],
                      "failure_tag": "RUN_ERROR" if r_c.get("_run_error") else tag_failure_decomposition(r_c, task)})
        if (task["id"], "peer_to_peer") not in done_keys:
            r_p = safe_run(run_p2p_decomposition, task, "peer-to-peer decomposition")
            save_row({"task_id": task["id"], "family": "decomposition", "condition": "peer_to_peer",
                      "agent_count": task["agent_count"], "correct": r_p["correct"],
                      "n_messages": r_p["n_messages"], "n_turns": r_p["n_turns"],
                      "n_tokens": r_p["n_tokens"], "wall_time": r_p["wall_time"],
                      "failure_tag": "RUN_ERROR" if r_p.get("_run_error") else tag_failure_decomposition(r_p, task)})

    for task in negotiation_tasks:
        if (task["id"], "centralised") not in done_keys:
            r_c = safe_run(run_centralised_negotiation, task, "centralised negotiation")
            save_row({"task_id": task["id"], "family": "negotiation", "condition": "centralised",
                      "agent_count": task["agent_count"], "correct": r_c["correct"],
                      "n_messages": r_c["n_messages"], "n_turns": r_c["n_turns"],
                      "n_tokens": r_c["n_tokens"], "wall_time": r_c["wall_time"],
                      "failure_tag": "RUN_ERROR" if r_c.get("_run_error") else tag_failure_negotiation(r_c),
                      "efficiency": r_c["efficiency"]})
        if (task["id"], "peer_to_peer") not in done_keys:
            r_p = safe_run(run_p2p_negotiation, task, "peer-to-peer negotiation")
            save_row({"task_id": task["id"], "family": "negotiation", "condition": "peer_to_peer",
                      "agent_count": task["agent_count"], "correct": r_p["correct"],
                      "n_messages": r_p["n_messages"], "n_turns": r_p["n_turns"],
                      "n_tokens": r_p["n_tokens"], "wall_time": r_p["wall_time"],
                      "failure_tag": "RUN_ERROR" if r_p.get("_run_error") else tag_failure_negotiation(r_p),
                      "efficiency": r_p["efficiency"]})

    csv_file.close()

    groups = {}
    for row in rows:
        key = (row["family"], row["condition"], row["agent_count"])
        groups.setdefault(key, []).append(row)
    summary_rows = []
    for (family, condition, ac), rs in sorted(groups.items()):
        n = len(rs)
        accuracy = sum(1 for r in rs if r["correct"]) / n
        avg_msgs = stats.mean(r["n_messages"] for r in rs)
        avg_turns = stats.mean(r["n_turns"] for r in rs)
        avg_tokens = stats.mean(r["n_tokens"] for r in rs)
        avg_time_ms = stats.mean(r["wall_time"] for r in rs) * 1000
        summary_rows.append({"family": family, "condition": condition, "agent_count": ac, "n_runs": n,
                              "accuracy": round(accuracy, 3), "avg_messages": round(avg_msgs, 2),
                              "avg_turns": round(avg_turns, 2), "avg_tokens": round(avg_tokens, 1),
                              "avg_wall_time_ms": round(avg_time_ms, 3),
                              "comm_overhead_per_agent": round(avg_msgs / ac, 2)})
    with open("results_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "condition", "agent_count", "n_runs", "accuracy",
                                           "avg_messages", "avg_turns", "avg_tokens", "avg_wall_time_ms",
                                           "comm_overhead_per_agent"])
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Stage 1 complete: {len(rows)} total runs. results_raw.csv / results_summary.csv written.")
    return rows, summary_rows


# =====================================================================
# SECTION 7 — STAGE 2: SENSITIVITY ANALYSIS (originally sensitivity_experiment.py)
# =====================================================================

N_SEEDS = 8
SEEDS = list(range(100, 100 + N_SEEDS))
ERROR_RATIOS = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]


def _run_chain_for_sensitivity(task, rng, error_rate):
    value = task["seed_value"]
    for rule in task["rules"]:
        value, _ = agent_step(value, rule, error_rate, rng)
    return abs(value - task["answer"]) < 1e-6


def run_sensitivity_analysis():
    tasks = build_gsm8k_chain_tasks(per_bucket=15, seed=32)
    er_rows = []
    for ratio in ERROR_RATIOS:
        p2p_rate = CENTRALISED_ERROR_RATE * ratio
        c_accs, p_accs = [], []
        for seed in SEEDS:
            rng = random.Random(seed)
            c_correct = sum(_run_chain_for_sensitivity(t, rng, CENTRALISED_ERROR_RATE) for t in tasks)
            p_correct = sum(_run_chain_for_sensitivity(t, rng, p2p_rate) for t in tasks)
            c_accs.append(c_correct / len(tasks))
            p_accs.append(p_correct / len(tasks))
        mean_c, mean_p = stats.mean(c_accs), stats.mean(p_accs)
        er_rows.append({"p2p_error_rate_ratio": ratio, "p2p_error_rate": round(p2p_rate, 4),
                         "centralised_accuracy_mean": round(mean_c, 4), "centralised_accuracy_std": round(stats.stdev(c_accs), 4),
                         "p2p_accuracy_mean": round(mean_p, 4), "p2p_accuracy_std": round(stats.stdev(p_accs), 4),
                         "accuracy_gap": round(mean_c - mean_p, 4)})
    with open("sensitivity_error_rate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(er_rows[0].keys()))
        w.writeheader()
        w.writerows(er_rows)

    decomp_tasks = build_gsm8k_chain_tasks(per_bucket=15, seed=32)
    neg_tasks = build_dond_negotiation_tasks(per_bucket=15, seed=32)
    results = {("decomposition", "centralised"): [], ("decomposition", "peer_to_peer"): [],
               ("negotiation", "centralised"): [], ("negotiation", "peer_to_peer"): []}
    for seed in SEEDS:
        rng = random.Random(seed)
        results[("decomposition", "centralised")].append(sum(run_centralised_decomposition(t, rng)["correct"] for t in decomp_tasks) / len(decomp_tasks))
        results[("decomposition", "peer_to_peer")].append(sum(run_p2p_decomposition(t, rng)["correct"] for t in decomp_tasks) / len(decomp_tasks))
        results[("negotiation", "centralised")].append(sum(run_centralised_negotiation(t, rng)["correct"] for t in neg_tasks) / len(neg_tasks))
        results[("negotiation", "peer_to_peer")].append(sum(run_p2p_negotiation(t, rng)["correct"] for t in neg_tasks) / len(neg_tasks))
    seed_rows = []
    for (family, condition), accs in results.items():
        seed_rows.append({"family": family, "condition": condition, "n_seeds": len(accs),
                           "accuracy_mean": round(stats.mean(accs), 4), "accuracy_std": round(stats.stdev(accs), 4),
                           "accuracy_min": round(min(accs), 4), "accuracy_max": round(max(accs), 4)})
    with open("sensitivity_seeds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seed_rows[0].keys()))
        w.writeheader()
        w.writerows(seed_rows)

    print("Stage 2 complete: sensitivity_error_rate.csv / sensitivity_seeds.csv written.")
    return er_rows, seed_rows


# =====================================================================
# SECTION 8 — STAGE 3: FIX COMPARISON (originally run_fix_comparison.py)
# =====================================================================

def tag_failure_decomposition_fix(result, task):
    if result["correct"]:
        return None
    diff = result["final_value"] - task["answer"]
    if abs(diff) <= 3:
        return "Information Loss (single mis-transmitted value)"
    return "Compounding Information Loss (multiple mis-transmitted values)"


def tag_failure_negotiation_fix(result):
    if result["correct"]:
        return None
    if result.get("non_convergence"):
        return "Non-Convergence (no agreement reached within round limit)"
    return "Inefficient Allocation (agreement reached, but below efficiency threshold)"


def run_fix_comparison():
    decomposition_tasks = build_gsm8k_chain_tasks(per_bucket=TASKS_PER_BUCKET, seed=MASTER_SEED)
    negotiation_tasks = build_dond_negotiation_tasks(per_bucket=TASKS_PER_BUCKET, seed=MASTER_SEED)

    rng_c = random.Random(MASTER_SEED)
    rng_before = random.Random(MASTER_SEED)
    rng_after = random.Random(MASTER_SEED)

    rows = []
    for task in decomposition_tasks:
        r_c = run_centralised_decomposition(task, rng_c)
        rows.append({"task_id": task["id"], "family": "decomposition", "condition": "centralised",
                      "agent_count": task["agent_count"], "correct": r_c["correct"],
                      "n_messages": r_c["n_messages"], "n_tokens": r_c["n_tokens"],
                      "failure_tag": tag_failure_decomposition_fix(r_c, task)})
        r_before = run_p2p_decomposition(task, rng_before)
        rows.append({"task_id": task["id"], "family": "decomposition", "condition": "peer_to_peer_BEFORE",
                      "agent_count": task["agent_count"], "correct": r_before["correct"],
                      "n_messages": r_before["n_messages"], "n_tokens": r_before["n_tokens"],
                      "failure_tag": tag_failure_decomposition_fix(r_before, task)})
        r_after = run_p2p_decomposition_fixed(task, rng_after)
        rows.append({"task_id": task["id"], "family": "decomposition", "condition": "peer_to_peer_AFTER",
                      "agent_count": task["agent_count"], "correct": r_after["correct"],
                      "n_messages": r_after["n_messages"], "n_tokens": r_after["n_tokens"],
                      "failure_tag": tag_failure_decomposition_fix(r_after, task)})

    for task in negotiation_tasks:
        r_c = run_centralised_negotiation(task, rng_c)
        rows.append({"task_id": task["id"], "family": "negotiation", "condition": "centralised",
                      "agent_count": task["agent_count"], "correct": r_c["correct"],
                      "n_messages": r_c["n_messages"], "n_tokens": r_c["n_tokens"],
                      "failure_tag": tag_failure_negotiation_fix(r_c)})
        r_before = run_p2p_negotiation(task, rng_before)
        rows.append({"task_id": task["id"], "family": "negotiation", "condition": "peer_to_peer_BEFORE",
                      "agent_count": task["agent_count"], "correct": r_before["correct"],
                      "n_messages": r_before["n_messages"], "n_tokens": r_before["n_tokens"],
                      "failure_tag": tag_failure_negotiation_fix(r_before)})
        r_after = run_p2p_negotiation_fixed(task, rng_after)
        rows.append({"task_id": task["id"], "family": "negotiation", "condition": "peer_to_peer_AFTER",
                      "agent_count": task["agent_count"], "correct": r_after["correct"],
                      "n_messages": r_after["n_messages"], "n_tokens": r_after["n_tokens"],
                      "failure_tag": tag_failure_negotiation_fix(r_after)})

    fieldnames = ["task_id", "family", "condition", "agent_count", "correct", "n_messages", "n_tokens", "failure_tag"]
    with open("fix_comparison_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    groups = {}
    for row in rows:
        groups.setdefault((row["family"], row["condition"]), []).append(row)
    summary = []
    for (family, condition), rs in sorted(groups.items()):
        n = len(rs)
        acc = sum(1 for r in rs if r["correct"]) / n
        avg_msgs = stats.mean(r["n_messages"] for r in rs)
        avg_tokens = stats.mean(r["n_tokens"] for r in rs)
        summary.append({"family": family, "condition": condition, "n_runs": n,
                         "accuracy": round(acc, 4), "avg_messages": round(avg_msgs, 2), "avg_tokens": round(avg_tokens, 1)})
    with open("fix_comparison_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "condition", "n_runs", "accuracy", "avg_messages", "avg_tokens"])
        w.writeheader()
        w.writerows(summary)

    print(f"Stage 3 complete: {len(rows)} total runs. fix_comparison_raw.csv / fix_comparison_summary.csv written.")
    return rows, summary


# =====================================================================
# SECTION 9 — STAGE 4: CHARTS (originally make_charts.py)
# =====================================================================

CENTRAL_COLOR = "#2E74B5"
P2P_COLOR = "#C00000"
AFTER_COLOR = "#548235"


def chart1_main_accuracy():
    rows = list(csv.DictReader(open("results_summary.csv")))
    families = ["decomposition", "negotiation"]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(families))
    width = 0.35

    def family_accuracy(family, condition):
        rs = [r for r in rows if r["family"] == family and r["condition"] == condition]
        n = sum(int(r["n_runs"]) for r in rs)
        correct = sum(float(r["accuracy"]) * int(r["n_runs"]) for r in rs)
        return correct / n * 100

    central_vals = [family_accuracy(f, "centralised") for f in families]
    p2p_vals = [family_accuracy(f, "peer_to_peer") for f in families]
    b1 = ax.bar(x - width / 2, central_vals, width, label="Centralised", color=CENTRAL_COLOR)
    b2 = ax.bar(x + width / 2, p2p_vals, width, label="Peer-to-Peer", color=P2P_COLOR)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in families])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Main Experiment: Accuracy by Coordination Structure", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("chart1_main_accuracy.png")
    plt.close()
    print("Saved chart1_main_accuracy.png")


def chart2_main_heatmap():
    rows = list(csv.DictReader(open("results_summary.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plt.subplots_adjust(wspace=0.5)
    families = ["decomposition", "negotiation"]
    conditions = ["centralised", "peer_to_peer"]
    labels = ["Centralised", "Peer-to-Peer"]

    for ax, family in zip(axes, families):
        frows = [r for r in rows if r["family"] == family]
        agent_counts = sorted(set(int(r["agent_count"]) for r in frows))
        grid = np.zeros((2, len(agent_counts)))
        for i, cond in enumerate(conditions):
            for j, ac in enumerate(agent_counts):
                r = next(x for x in frows if x["condition"] == cond and int(x["agent_count"]) == ac)
                grid[i, j] = float(r["accuracy"]) * 100
        im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(agent_counts)))
        ax.set_xticklabels([f"{a}" for a in agent_counts])
        ax.set_yticks(range(2))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Number of agents")
        for i in range(2):
            for j in range(len(agent_counts)):
                val = grid[i, j]
                color = "white" if val < 50 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=color, fontweight="bold")
        ax.set_title(family.capitalize(), fontweight="bold")
    fig.colorbar(im, ax=axes, label="Accuracy (%)", shrink=0.8)
    fig.suptitle("Main Experiment: Accuracy by Team Size", fontweight="bold", y=1.03)
    plt.savefig("chart2_main_heatmap.png", bbox_inches="tight")
    plt.close()
    print("Saved chart2_main_heatmap.png")


def chart3_main_overhead():
    rows = list(csv.DictReader(open("results_summary.csv")))
    families = ["decomposition", "negotiation"]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(families))
    width = 0.35

    def family_messages(family, condition):
        rs = [r for r in rows if r["family"] == family and r["condition"] == condition]
        n = sum(int(r["n_runs"]) for r in rs)
        total = sum(float(r["avg_messages"]) * int(r["n_runs"]) for r in rs)
        return total / n

    central_vals = [family_messages(f, "centralised") for f in families]
    p2p_vals = [family_messages(f, "peer_to_peer") for f in families]
    b1 = ax.bar(x - width / 2, central_vals, width, label="Centralised", color=CENTRAL_COLOR)
    b2 = ax.bar(x + width / 2, p2p_vals, width, label="Peer-to-Peer", color=P2P_COLOR)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15, f"{h:.1f}", ha="center", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in families])
    ax.set_ylabel("Avg. messages per task")
    ax.set_title("Main Experiment: Communication Overhead", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("chart3_main_overhead.png")
    plt.close()
    print("Saved chart3_main_overhead.png")


def chart4_fix_accuracy():
    rows = list(csv.DictReader(open("fix_comparison_summary.csv")))
    families = ["decomposition", "negotiation"]
    conditions = ["centralised", "peer_to_peer_BEFORE", "peer_to_peer_AFTER"]
    labels = ["Centralised", "P2P\n(BEFORE)", "P2P\n(AFTER)"]
    colors = [CENTRAL_COLOR, P2P_COLOR, AFTER_COLOR]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, family in zip(axes, families):
        frows = [r for r in rows if r["family"] == family]
        values = [float(next(r for r in frows if r["condition"] == c)["accuracy"]) * 100 for c in conditions]
        bars = ax.bar(labels, values, color=colors, width=0.6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 110)
        ax.set_title(family.capitalize(), fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Before vs After the Fix", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig("chart4_fix_accuracy.png")
    plt.close()
    print("Saved chart4_fix_accuracy.png")


def chart5_fix_failures():
    rows = list(csv.DictReader(open("fix_comparison_raw.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    families = ["decomposition", "negotiation"]
    for ax, family in zip(axes, families):
        before = Counter(r["failure_tag"] for r in rows if r["family"] == family and r["condition"] == "peer_to_peer_BEFORE" and r["failure_tag"])
        after = Counter(r["failure_tag"] for r in rows if r["family"] == family and r["condition"] == "peer_to_peer_AFTER" and r["failure_tag"])
        tags = sorted(set(before) | set(after))
        short = [t.split(" (")[0] for t in tags]
        x = np.arange(len(tags))
        width = 0.35
        b1 = ax.bar(x - width / 2, [before.get(t, 0) for t in tags], width, label="BEFORE", color=P2P_COLOR)
        b2 = ax.bar(x + width / 2, [after.get(t, 0) for t in tags], width, label="AFTER", color=AFTER_COLOR)
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, str(int(h)), ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=15, ha="right")
        ax.set_ylabel("Number of failures")
        ax.set_title(family.capitalize(), fontweight="bold")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Failure Types: Before vs After the Fix", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig("chart5_fix_failures.png")
    plt.close()
    print("Saved chart5_fix_failures.png")


def chart6_fix_overhead():
    rows = list(csv.DictReader(open("fix_comparison_summary.csv")))
    families = ["decomposition", "negotiation"]
    conditions = ["centralised", "peer_to_peer_BEFORE", "peer_to_peer_AFTER"]
    labels = ["Centralised", "P2P\n(BEFORE)", "P2P\n(AFTER)"]
    colors = [CENTRAL_COLOR, P2P_COLOR, AFTER_COLOR]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, family in zip(axes, families):
        frows = [r for r in rows if r["family"] == family]
        values = [float(next(r for r in frows if r["condition"] == c)["avg_messages"]) for c in conditions]
        bars = ax.bar(labels, values, color=colors, width=0.6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", fontweight="bold")
        ax.set_ylabel("Avg. messages per task")
        ax.set_title(family.capitalize(), fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Communication Overhead: Before vs After the Fix", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig("chart6_fix_overhead.png")
    plt.close()
    print("Saved chart6_fix_overhead.png")


def run_all_charts():
    chart1_main_accuracy()
    chart2_main_heatmap()
    chart3_main_overhead()
    chart4_fix_accuracy()
    chart5_fix_failures()
    chart6_fix_overhead()
    print("Stage 4 complete: all 6 charts written.")


# =====================================================================
# MAIN — run all four stages in order
# =====================================================================

if __name__ == "__main__":
    print("=== STAGE 1: Main experiment ===")
    run_main_experiment()

    print("\n=== STAGE 2: Sensitivity analysis ===")
    run_sensitivity_analysis()

    print("\n=== STAGE 3: Fix comparison ===")
    run_fix_comparison()

    print("\n=== STAGE 4: Charts ===")
    run_all_charts()

    print("\nAll stages complete.")
