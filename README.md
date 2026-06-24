# EMS-DT: A Digital Twin-Assisted Emergency Medical System

A reinforcement-learning dispatch engine and decision-support layer for ambulance-and-hospital assignment, built as a digital twin of the MEMS 108 emergency medical service for Nashik district, Maharashtra.
 This repository contains the simulation environment, training pipelines, and supervisory decision-support agent described in the thesis.

## The problem

In India, an estimated ~30% of emergency deaths are linked to delays in care, not unavailability of it. Maharashtra's MEMS 108 service — 937 ambulances, 24/7, free, over 1.5 crore calls handled since launch — still dispatches reactively: a human operator decides which ambulance to send and which hospital to send it to under uncertainty, with no systematic way to account for live traffic, hospital bed occupancy, or how long a patient has already been waiting.

EMS-DT builds a calibrated, district-scale simulation of that decision and trains a reinforcement-learning agent to make it — jointly optimizing **which ambulance** and **which hospital**, not just one or the other, under the same constraints a real dispatcher faces.

## What's in this repo

| File | Role |
|---|---|
| `rl_train.py` | Core `EMSEnv` — a Gymnasium-compatible discrete-event simulation of incident arrival, ambulance dispatch, hospital admission, and patient discharge for Nashik district (46 ambulances, 155 hospitals). Defines the action mask, the 7-component reward function, and the synthetic incident generator. |
| `ems_candidate_env.py` | `CandidateActionEnv` — wraps the environment's full 7,130-action space into a fixed, always-valid `Discrete(k)` candidate shortlist so off-policy methods without native action-masking can be trained on the same task. |
| `custom_rl_agents.py` | From-scratch PyTorch implementations of DQN, Double DQN, Dueling DQN, and Discrete Soft Actor-Critic, sharing one replay buffer implementation for an apples-to-apples comparison. |
| `train_ppo_ems.py` | Production training pipeline for **MaskablePPO** (sb3-contrib) directly on the full masked 7,130-action space, with VecNormalize, checkpointing, dispatch-level logging, and evaluation/plotting. |
| `train_multi_algo_ems.py` | Shared orchestrator for DQN / Double DQN / Dueling DQN / Discrete SAC / a plain-PPO control, all trained against the candidate-action wrapper. |
| `train_dqn_ems.py`, `train_discrete_sac_ems.py` | One-line CLI convenience wrappers around the orchestrator above. |
| `decision_support_agent.py` | Supervisory agent that loads a trained policy, independently re-validates and re-ranks every feasible alternative against ground-truth reward, attaches plain-language risk flags, and generates a human-readable explanation (LLM-assisted with a deterministic fallback). |

## How the dispatch decision is modeled

- **State** — a 4,421-dimensional vector: the waiting-incident queue (top 30, severity- and age-ranked), all 46 ambulances' status and location, all 155 hospitals' specialty and bed-occupancy data across 4 bed types and 15 specialties, plus global context (time, traffic).
- **Action** — `Discrete(46 × 155 = 7,130)`: pick one ambulance, one destination hospital.
- **Masking** — a boolean feasibility mask recomputed every decision point (ambulance idle, hospital offers the needed specialty and has a free bed of the severity-appropriate type), pruned to the nearest 6/4/3 ambulances and 8/6/4 hospitals by severity, with a hard rule reserving ALS units for high-severity calls whenever one is available.
- **Reward** — a weighted sum of seven components computed at every dispatch:

  ```
  R = 1.0·match + 1.0·travel + 1.5·eta_gap + 1.0·golden_hour + 1.0·wait + 0.3·load + 0.1·opportunity_cost
  ```

  encoding ALS/BLS-severity matching, log-scaled travel-time cost, relative efficiency versus the best feasible option, a hard 60-minute golden-hour threshold, saturating wait-time pressure, hospital load-balancing, and ambulance-coverage opportunity cost.

## Algorithm comparison

Five RL algorithms were trained for 500 episodes each under an identical reward function and action-candidate structure: **PPO**, **A2C**, **Dueling DQN**, **Rainbow DQN**, and **Discrete SAC**.

| Algorithm | Last-50-episode adjusted reward mean |
|---|---|
| **Discrete SAC** | **533.78** |
| Dueling DQN | 384.12 |
| Rainbow DQN | 212.81 |
| PPO | −127.54 |
| A2C | −300.14 |

Discrete SAC ranked first across every reward statistic tracked (final-episode, smoothed-trend, and last-50-window mean), consistent with its entropy-regularized exploration and twin-critic overestimation control being particularly well-suited to a small, distance-ranked candidate set where several options look similar at any given decision point. Full curves and per-algorithm analysis are in the thesis, Chapter 7.

## Decision-support layer

A trained policy's raw action isn't enough for a safety-relevant recommendation, so `decision_support_agent.py` wraps it with:
- independent re-validation against the live environment's own feasibility rules (not the network's prediction),
- a full ranking of every feasible alternative by ground-truth reward, so a recommendation's rank and reward gap vs. the best option are always known,
- plain-language risk flags (BLS dispatched to a high-severity case, golden-hour miss, ETA significantly worse than the best alternative, destination hospital near capacity),
- an LLM-generated explanation for human dispatchers — used strictly as a prose layer over already-validated, already-ranked structured data, with a deterministic template fallback if no LLM endpoint is configured.

## Tech stack

Python · PyTorch · Gymnasium · Stable-Baselines3 / sb3-contrib (MaskablePPO) · pandas / NumPy · OpenStreetMap + OSMnx (road-network-aware travel-time estimation) · LangChain + Hugging Face (decision-support explanation layer)

## Status and limitations

This is a research prototype validated on synthetic incident streams calibrated against real MEMS 108 historical data for Nashik district — it has not been field-tested. Known open items: district-specific calibration (would need re-running the pipeline for other districts), hospital occupancy is registry-estimated rather than live, ETA is a traffic-multiplier model rather than full road-network simulation, reward weights reflect design priorities rather than validated clinical consensus, and the current comparison is reward-based only (no reported high-severity-specific latency or worst-case robustness metrics yet).


