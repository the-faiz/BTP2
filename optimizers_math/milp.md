# MILP Optimizer: Mathematical Formulation

This solver uses a Mixed-Integer Linear Program (MILP) to optimize user admission, slice assignment, and PRB allocation.

**Sets and indices**
- Users: `i ∈ U`
- Slices: `k ∈ K`
- Tiers: `t ∈ T`, with a tier function `tier(i)`

**Parameters**
- `R_i`: target rate for user `i` (from tier)
- `w_i`: weight for user `i` (from tier)
- `P_i`: price paid by user `i` (from tier)
- `C_k`: PRB capacity of slice `k`
- `c_k`: cost per PRB in slice `k`
- `e_k`: efficiency (bandwidth per PRB) of slice `k`
- `θ`: satisfaction threshold (default `0.8`)
- `ε`: small slack for logical linking
- `m`: minimum PRBs per admitted user
- `N_t`: minimum number of satisfied users required in tier `t`

**Decision variables**
- `x_{i,k} ∈ {0,1}`: 1 if user `i` is assigned to slice `k`
- `p_{i,k} ≥ 0`: PRBs allocated to user `i` in slice `k`
- `r_i ≥ 0`: allocated rate for user `i`
- `s_i ∈ [0,1]`: satisfaction of user `i`
- `y_i ∈ {0,1}`: 1 if user `i` is satisfied

**Allocated rate and satisfaction**
- Allocated rate: `r_i = Σ_k p_{i,k} * e_k`
- Satisfaction upper bound: `s_i ≤ r_i / R_i`, `s_i ≤ 1`

**Objective**
Maximize penalized weighted satisfaction plus profit:
```
max  Σ_i (w_i * s_i) + Σ_i Σ_k (x_{i,k} * P_i) - Σ_i Σ_k (c_k * p_{i,k})
```

**Constraints**
1. At most one slice per user:
```
Σ_k x_{i,k} ≤ 1,  ∀i
```
2. Slice capacity:
```
Σ_i p_{i,k} ≤ C_k,  ∀k
```
3. Coupling (PRBs only if assigned):
```
p_{i,k} ≤ C_k * x_{i,k},  ∀i,k
```
4. Minimum PRBs if admitted:
```
Σ_k p_{i,k} ≥ m * Σ_k x_{i,k},  ∀i
```
5. Satisfaction threshold linking:
```
s_i - θ * y_i ≥ 0
s_i - (θ - ε) * y_i ≤ θ - ε
```
6. Minimum satisfied users per tier:
```
Σ_{i: tier(i)=t} y_i ≥ N_t,  ∀t
```

**Notes**
- This is a linear objective with linear constraints and integrality on `x_{i,k}`, `y_i`.
- Solved using `scipy.optimize.milp` (HiGHS).
