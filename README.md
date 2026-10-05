# Multi-Agent Collaboration Systems for AI — Group 32

A controlled simulation study comparing **centralised orchestration** (a controller agent manages the team) against **peer-to-peer negotiation** (agents coordinate directly, no controller) in small LLM-style agent teams of 2–5 agents.

This repository contains the full, reproducible experiment: real datasets, simulation code, a sensitivity/robustness analysis, two targeted fixes with a before/after comparison, and 6 charts — all runnable from a single Python file.

---

## The research question

> Between centralised orchestration and peer-to-peer negotiation, which coordination structure performs better for small LLM agent teams — and under what conditions?

The literature reviewed for this project (AutoGen, BOLAA, MetaGPT, ChatDev for centralised; Hua et al. 2024 for peer-to-peer) never compares the two coordination structures under identical, matched conditions. This project closes that gap with a controlled simulation.

---

## How agents are simulated

**No live LLM API is used anywhere in this repository.** Each agent's reasoning step is a deterministic rule-based policy with a fixed, literature-informed error rate, not a call to an external model. This isolates coordination *structure* as the variable under test, independent of any particular model's behaviour, cost, or availability. See [Limitations](#limitations) below.

---

## Datasets used

Two real, independent, publicly published datasets — no invented numbers.

| Dataset | Used for | Source | Licence |
|---|---|---|---|
| **GSM8K** (Cobbe et al., 2021) | Decomposition / collaborative-reasoning tasks | [github.com/openai/grade-school-math](https://github.com/openai/grade-school-math) | MIT |
| **Deal-or-No-Deal corpus** (Lewis et al., 2017) | Negotiation tasks | [github.com/facebookresearch/end-to-end-negotiator](https://github.com/facebookresearch/end-to-end-negotiator) | CC BY-NC 4.0 |

GSM8K's step-by-step calculator annotations are parsed into genuine multi-hop reasoning chains, one hop per agent. Deal-or-No-Deal's real private human valuations are used directly as agents' negotiation preferences; building 3–5 agent scenarios (the original corpus is bilateral) is done by pooling multiple real valuation vectors that share an identical item-count vector — the grouping is ours, every valuation number is real.

---

## Repository structure

```
.
├── full_experiment.py      # the entire project — run this one file
├── real_data/
│   ├── gsm8k_test.jsonl    # GSM8K test set (1,319 problems)
│   └── dond_test.txt       # Deal-or-No-Deal test set (1,052 sessions)
└── README.md
```

Running `full_experiment.py` generates everything else automatically:

```
results_raw.csv                 # Stage 1 output: per-run results, main experiment
results_summary.csv             # Stage 1 output: aggregated results
sensitivity_error_rate.csv      # Stage 2 output: accuracy vs assumed error-rate ratio
sensitivity_seeds.csv           # Stage 2 output: accuracy mean/std across 8 random seeds
fix_comparison_raw.csv          # Stage 3 output: per-run before/after results
fix_comparison_summary.csv      # Stage 3 output: aggregated before/after results
chart1_main_accuracy.png        # Stage 4 output: accuracy, centralised vs peer-to-peer
chart2_main_heatmap.png         # Stage 4 output: accuracy by team size (2–5 agents)
chart3_main_overhead.png        # Stage 4 output: communication overhead
chart4_fix_accuracy.png         # Stage 4 output: accuracy before vs after the fix
chart5_fix_failures.png         # Stage 4 output: failure types before vs after the fix
chart6_fix_overhead.png         # Stage 4 output: communication overhead before vs after the fix
```

---

## How to run it (no API key required)

```bash
pip install matplotlib numpy
python full_experiment.py
```

That's it — one command runs all four stages in order and prints progress as it goes:

```
=== STAGE 1: Main experiment ===
Stage 1 complete: 234 total runs. results_raw.csv / results_summary.csv written.

=== STAGE 2: Sensitivity analysis ===
Stage 2 complete: sensitivity_error_rate.csv / sensitivity_seeds.csv written.

=== STAGE 3: Fix comparison ===
Stage 3 complete: 351 total runs. fix_comparison_raw.csv / fix_comparison_summary.csv written.

=== STAGE 4: Charts ===
Stage 4 complete: all 6 charts written.
```

Stage 1 is resume-safe — if interrupted, re-running the script picks up from `results_raw.csv` rather than starting over or losing progress.

---

## What each stage does

**Stage 1 — Main experiment.** Builds 234 matched tasks from the two real datasets (2–5 agents, both coordination structures) and runs them, logging accuracy, messages, tokens, wall time, and failure type for every run.

**Stage 2 — Sensitivity analysis.** Tests whether the main result depends on one specific assumption (that peer-to-peer agents are 3× more error-prone than centralised agents) by sweeping that assumption from 1× to 6×, across 8 random seeds, and checking whether the accuracy gap survives.

**Stage 3 — Fix comparison.** Diagnoses the two dominant failure modes from Stage 1, implements a targeted code fix for each, and re-runs the full comparison to measure before vs after.

**Stage 4 — Charts.** Generates 6 charts from the CSVs produced above: 3 for the main experiment, 3 for the fix results.

---

## Key findings

### 1. Main comparison
Centralised orchestration outperformed peer-to-peer on accuracy in both task families:

| Family | Centralised | Peer-to-peer |
|---|---|---|
| Decomposition (GSM8K) | 94.7% | 71.9% |
| Negotiation (Deal-or-No-Deal) | 96.7% | 30.0% |

Communication overhead's direction was not consistent — peer-to-peer used fewer messages at low agent counts but *more* at high agent counts in the negotiation family.

### 2. Sensitivity analysis — the result is conditional, not absolute
When centralised and peer-to-peer agents are given **equal** reliability, their accuracy is statistically indistinguishable (94.1% vs 93.0%). Centralisation's advantage only emerges, and grows, as peer-to-peer's assumed unreliability increases relative to centralised — reaching a 31.6-point gap at 6× the error rate.

### 3. Two targeted fixes, two different outcomes
- **Decomposition fix (self-check before transmitting):** accuracy rose from 70.2% → 91.2%, closing the entire gap and even surpassing centralised, using *fewer* messages.
- **Negotiation fix (require a full consensus pass before stopping):** accuracy stayed flat at 30.0% despite more than doubling the message count — revealing that premature stopping was not negotiation's real bottleneck. The actual bottleneck is the trade-acceptance rule itself, which does not account for global allocation quality.

This negative result is reported deliberately rather than hidden: it narrows down the real cause of peer-to-peer negotiation's weakness more precisely than the original diagnosis did.

---

## Limitations

- **No live LLM is used.** Agent reasoning is a deterministic simulation with literature-informed error rates, not a call to a real language model. This isolates coordination structure as a variable but does not demonstrate how real LLM agents would behave.
- The error-rate asymmetry between conditions is an assumption drawn from a qualitative finding in the literature (Liu et al., 2023), not a measured value — which is exactly what the sensitivity analysis in this repository tests directly.
- Deal-or-No-Deal is natively bilateral; 3–5 agent negotiation scenarios are constructed by this project by pooling multiple real sessions' valuations, not sourced from a native multi-party negotiation dataset.
- The failure-tag categories used here are a simplified, rule-based stand-in for the MAST taxonomy (Cemri et al., 2025), not the validated 94%-accuracy LLM classifier described in that paper.

---

## References

- Cobbe, K. et al. (2021) *Training Verifiers to Solve Math Word Problems*. arXiv:2110.14168.
- Lewis, M. et al. (2017) *Deal or No Deal? End-to-End Learning of Negotiation Dialogues*. EMNLP 2017.
- Liu, Z. et al. (2023) *BOLAA: Benchmarking and Orchestrating LLM-augmented Autonomous Agents*. arXiv:2308.05960.
- Wu, Q. et al. (2023) *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework*. arXiv:2308.08155.
- Hua, W. et al. (2024) *Game-Theoretic LLM: Agent Workflow for Negotiation Games*. arXiv:2411.05990.
- Cemri, M. et al. (2025) *Why Do Multi-Agent LLM Systems Fail?* NeurIPS 2025.
