import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from policies.lsm import LSMPolicy
from policies.rpo import RPOPolicy
from section51_gbm_comparison.gbm_trajectories import generate_gbm_trajectories, graph_gbm_trajectories
from plot_boundaries import plot_boundaries, plot_rpo_boundaries

import matplotlib.pyplot as plt
from pathlib import Path

# Payoff for an American call
K = 100.0
def payoff(t, x):
    return max(x - K, 0)

basis_functions = [
    lambda x: 1.0,                            # Constant term (bias/intercept)
    lambda x: (x - K) / (x + 1e-5),           # Scaled payoff (zero near strike, increasing with moneyness)
    lambda x: ((x - K) / (x + 1e-5))**2,      # Quadratic in scaled payoff — allows curvature
    lambda x: np.exp(-np.abs(x - K) / K),    # Exponential decay around strike (sharp transition modeling)
]

gmbs = []
for S0 in range (80, 120, 5):
    for mu in np.arange(-0.005, 0.005, 0.001):
        for sigma in np.arange(0.001, 0.01, 0.001):
            gmbs.append((S0, mu, sigma))

all_trajectories = [generate_gbm_trajectories(S0, mu, sigma, 100, 100, 5) for (S0, mu, sigma) in gmbs]
trajectories = np.concatenate(all_trajectories, axis=0)
graph_gbm_trajectories(trajectories, 100, 100, "test6_x4/gbm_trajectories.png")
print(f"Generated {len(trajectories)} trajectories.")

lsm_policy = LSMPolicy(
    basis_functions=basis_functions,
    payoff_fn=payoff,
    T=100,
)
lsm_policy.train(trajectories)
lsm_weights = lsm_policy.get_weights()

print("Trained LSM policy.")

rpo_lsm_weights = RPOPolicy(
    basis_functions=basis_functions,
    payoff_fn=payoff,
    T=100,
    steps=100,
    lr=1e-1,
    inner_epochs=50,
    outer_iterations=30,
    start_weights=lsm_weights
)
rpo_lsm_weights.train(trajectories, save_weights=True, save_path="test6_x4/rpo_weights.pt")
print("Trained RPO policies.")

# Plotting the boundaries
plot_boundaries(
    policies=[("LSM", lsm_policy)],
    t_values=list(range(100)),
    x_grid=np.linspace(50, 500, 700),
    title="LSM Boundary",
    filename="test6_x4/lsm_boundary.png"
)
print("LSM boundary plotted.")
plot_rpo_boundaries(
    policies=[("RPO LSM Weights", rpo_lsm_weights)],
    t_values=list(range(100)),
    x_grid=np.linspace(50, 500, 700),
    target_probs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    title="RPO Boundaries",
    filename="test6_x4/rpo_boundaries.png"
)
print("RPO boundaries plotted.")