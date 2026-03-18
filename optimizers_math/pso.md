# Particle Swarm Optimization (PSO)

## Overview
PSO is a population-based heuristic where each particle represents a candidate solution.
Particles move through the search space using their velocity, personal best, and global best.

## State
For particle i in dimension d:
- Position: x_i[d]
- Velocity: v_i[d]
- Personal best position: pbest_i[d]
- Global best position: gbest[d]

## Update Equations
At each iteration:

v_i[d] = w * v_i[d]
         + c1 * r1 * (pbest_i[d] - x_i[d])
         + c2 * r2 * (gbest[d] - x_i[d])

x_i[d] = x_i[d] + v_i[d]

Where:
- w is inertia weight
- c1 is cognitive coefficient
- c2 is social coefficient
- r1, r2 ~ Uniform(0, 1)

## Constraints
Positions are clamped to bounds:
- x_i[d] in [lb_d, ub_d]
Velocities are clamped to:
- v_i[d] in [-vmax, vmax]

## Objective
Each particle is scored using the penalized objective:
objective = weighted_satisfaction_penalized + profit

Weighted satisfaction is penalized by tier-minimum violations:
weighted_satisfaction_penalized =
    weighted_satisfaction - (minimum_satisfied_penalty * min_satisfied_violation)
