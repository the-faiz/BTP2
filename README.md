# BTP2: Multi-Slice Resource Allocation and User Satisfaction Optimization

This project models resource allocation in a sliced ISP network with heterogeneous users (Gold/Silver/Bronze tiers).  
The objective is to maximize weighted user satisfaction while minimizing infrastructure cost.

## Problem Statement

Given a user set `U = {1, 2, ..., n}`, each user `i` has:
- Subscription tier (Gold/Silver/Bronze)
- Weight `w_i`
- Target data rate `R_i_target`
- Channel/environment conditions (distance, mobility, interference)

Tier definitions:
- Gold: `R_i_target = 20 Mbps`, `w_i = 3`
- Silver: `R_i_target = 10 Mbps`, `w_i = 2`
- Bronze: `R_i_target = 5 Mbps`, `w_i = 1`

Required bandwidth per user is derived from Shannon capacity:

`B_i = R_i_target / log2(1 + gamma_i)`

where `gamma_i` is user SINR.

## ISP Slice Model

The ISP has 3 slices with fixed PRB capacities and per-PRB costs:

| Parameter | Slice 1 | Slice 2 | Slice 3 |
|---|---:|---:|---:|
| Efficiency `e_k` (BW per PRB) | 10 | 20 | 40 |
| Capacity `C_k` (PRBs) | 100 | 50 | 30 |
| Cost per PRB `c_k` | 5 | 10 | 15 |

Bandwidth-product cost rule from the formulation:
- `Cost = K * Bandwidth * Power`
- Current assumption: equal user power

## Optimization Formulation

Decision variables:
- `x_i,k in {0,1}`: user `i` assigned to slice `k`
- `p_i,k >= 0`: PRBs allocated to user `i` on slice `k`

Objective (as specified):

`Max Z = sum_i (w_i * Sat_i) - lambda * sum_i sum_k (c_k * p_i,k)`

Satisfaction can be represented as unmet-demand penalty or fulfillment:
- Unmet demand: `Sat_i = max(0, B_i - Allotted_BW_i)` (to minimize)
- Fulfillment: `Sat_i = min(B_i, sum_k p_i,k * e_k)` (to maximize)

Core constraints:
- Assignment: `sum_k x_i,k = 1` for all users `i`
- Slice capacity: `sum_i p_i,k <= C_k` for each slice `k`
- Coupling: `p_i,k <= C_k * x_i,k`

## Current Code Structure

- `main.py`: orchestration entry point
- `user_profile.py`:
  - `User` dataclass
  - `User.generate_user_profile(...)` static method for synthetic user generation
- `channel_model.py`:
  - `Channel` dataclass for channel properties and SINR/path-loss/interference computations

Current implementation covers user/channel modeling and synthetic profile generation.  
The optimization solver for `(x_i,k, p_i,k)` is the next major step.

## How to Run

```bash
python3 main.py
```

Expected output: a preview table of generated user profiles with tier, target rate, distance, speed, SINR, and required bandwidth.

## Assumptions in Current Prototype

- OFDMA-style PRB abstraction is used
- Users are distributed uniformly in a circular cell
- Mobility is sampled from `{0, 5, 40}` km/h
- Interference increases with distance to cell edge (simplified model)

## Suggested Next Steps

1. Implement slice allocation optimizer (MILP or heuristic like BIPSO).
2. Add satisfaction/fairness metrics (including Jain's Fairness Index).
3. Evaluate operating cost vs. satisfaction trade-off (`lambda` sweep).
4. Add Monte Carlo simulations and service migration experiments.
