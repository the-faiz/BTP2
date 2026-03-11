# Lagrangian Optimizer: Mathematical Outline

This optimizer uses a Lagrangian relaxation of the MILP, focusing on the capacity constraints and solving per-user subproblems iteratively with multiplier updates.

**Start from the MILP**
We relax the slice-capacity constraints:
```
Σ_i p_{i,k} ≤ C_k,  ∀k
```

Introduce multipliers `λ_k ≥ 0` and move the constraints into the objective:
```
L(λ) = max_{x,p,r,s,y}  Σ_i (w_i s_i) + Σ_i Σ_k (x_{i,k} P_i) - Σ_i Σ_k (c_k p_{i,k})
      + Σ_k λ_k (C_k - Σ_i p_{i,k})
```

This can be written as:
```
L(λ) = Σ_k λ_k C_k + Σ_i max_{x_i, p_i, r_i, s_i, y_i} [ w_i s_i + Σ_k x_{i,k} P_i - Σ_k (c_k + λ_k) p_{i,k} ]
```

**Per-user subproblem (fixed λ)**
For each user `i` and each slice `k`:
```
value(i,k) = P_i + w_i * s_i(p) - (c_k + λ_k) * p
```
where `s_i(p) = min(1, (e_k * p) / R_i)`.

The candidate `p` values are evaluated at:
- `p = m` (minimum PRBs if admitted)
- `p = R_i / e_k` (full satisfaction point)

Choose the best `(k, p)` with positive value, otherwise leave unadmitted.

**Multiplier update (subgradient)**
Let `g_k = Σ_i p_{i,k} - C_k` be the capacity violation (subgradient).
Update:
```
λ_k ← max(0, λ_k + α_t * g_k)
```
with a decaying step size `α_t`.

**Feasibility repair**
The relaxed solution can violate capacities. A greedy repair drops the lowest value-per-PRB assignments on any overloaded slice until `Σ_i p_{i,k} ≤ C_k`.

**Tier satisfaction repair**
If tier minimum satisfied counts are violated, allocate PRBs to the least-satisfied users of that tier (when capacity allows) to meet `s_i ≥ θ`.

**Notes**
- This is a heuristic: it trades optimality guarantees for speed and simplicity.
- It is useful as a scalable baseline or a warm-start for MILP.
