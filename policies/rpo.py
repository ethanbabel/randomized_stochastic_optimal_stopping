import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from policies.stopping_policy import StoppingPolicy


class RPOPolicy(StoppingPolicy):
    def __init__(self, basis_functions, payoff_fn, T, steps, lr=1e-1, inner_epochs=10, outer_iterations=10, start_weights=None):
        """
        Initialize RPOPolicy

        Parameters:
        - basis_functions (list): List of functions to apply to x
        - payoff_fn (function): Payoff function g(t, x)
        - T (float): Terminal time
        - steps (int): Number of discrete time steps
        - lr (float): Learning rate
        - inner_epochs (int): Number of epochs for inner optimization
        - outer_iterations (int): Number of outer iterations for optimization
        - start_weights (list): Initial weights for the regression model (if None, initialized to zero)
        """
        self.basis_functions = basis_functions
        self.payoff_fn = payoff_fn
        self.T = T
        self.steps = steps
        self.lr = lr
        self.inner_epochs = inner_epochs
        self.outer_iterations = outer_iterations
        self.k = len(basis_functions)
        # self.weights = nn.Parameter(torch.zeros((steps, self.k), dtype=torch.float32))
        if start_weights is None:
            self.weights = nn.Parameter(torch.zeros((steps, self.k), dtype=torch.float32))
        else:
            if len(start_weights) != steps:
                raise ValueError(f"Expected {steps} weights, got {len(start_weights)}")
            if len(start_weights[0]) != self.k:
                raise ValueError(f"Expected {self.k} weights per time step, got {len(start_weights[0])}")
            self.weights = nn.Parameter(torch.tensor(start_weights, dtype=torch.float32))

    def _phi(self, x):
        # Use FloatTensor to preserve computation graph if input x is a tensor
        return torch.FloatTensor([f(x) for f in self.basis_functions])

    def train(self, trajectories, save_weights=False, save_path="rpo_weights.pt"):
        """
        Trains the randomized policy via backward optimization.
        Parameters:
        - trajectories (list): List of price paths
        """
        optimizer = optim.Adam([self.weights], lr=self.lr)
        num_paths = len(trajectories)

        for iter_num in range(self.outer_iterations):
            print(f"--- Outer Iteration {iter_num + 1}/{self.outer_iterations} ---")

            # Initialize continuation values with terminal payoffs
            continuation_values = torch.zeros(num_paths, dtype=torch.float32)
            for i, traj in enumerate(trajectories):
                continuation_values[i] = self.payoff_fn(self.steps, traj[-1])

            for t in reversed(range(1, self.steps)):
                phi = torch.stack([self._phi(traj[t]) for traj in trajectories])  # (N, k)

                # Compute probability that the trajectory has not been exercised up to time t
                with torch.no_grad():
                    p_t = torch.ones(num_paths, dtype=torch.float32)
                    for t_prime in range(1, t):
                        phi_prime = torch.stack([self._phi(traj[t_prime]) for traj in trajectories])
                        b_prime = self.weights[t_prime]
                        logits = phi_prime @ b_prime
                        # Only update p_t for in-the-money trajectories at t_prime
                        itm_mask = torch.tensor([self.payoff_fn(t_prime, traj[t_prime]) > 0 for traj in trajectories])
                        p_t[itm_mask] *= 1 - torch.sigmoid(logits[itm_mask])

                def compute_objective():
                    logits = phi @ self.weights[t]
                    sigma = torch.sigmoid(logits)
                    payoffs = torch.tensor([self.payoff_fn(t, traj[t]) for traj in trajectories], dtype=torch.float32)
                    cont = continuation_values

                    expected_rewards = p_t * (payoffs * sigma + cont * (1 - sigma))

                    # Mask to in-the-money only
                    itm_mask = payoffs > 0
                    if itm_mask.any():
                        expected_rewards = expected_rewards[itm_mask]
                        return -expected_rewards.mean()  # negative for gradient descent

                    # reg_strength = 1e-3
                    # reg_term = reg_strength * self.weights[t].norm()**2
                    # return -expected_rewards.mean() + reg_term  # negative for gradient descent, plus L2 reg
                    return -expected_rewards.mean() 

                expected_rewards = 0
                for _ in range(self.inner_epochs):  # Optimize weights for this time step
                    optimizer.zero_grad()
                    expected_rewards = compute_objective()
                    loss = expected_rewards
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([self.weights], max_norm=1.0)
                    optimizer.step()

                # print(f"Iteration {iter_num + 1}, Time step {t}, Expected reward: {-expected_rewards}, p_t: {p_t.mean().item()}, sigma: {torch.sigmoid(phi @ self.weights[t]).mean().item()}")
                # print(f"min_sigma: {torch.sigmoid(phi @ self.weights[t]).min().item()}, max_sigma: {torch.sigmoid(phi @ self.weights[t]).max().item()}avg payoff: {payoffs.mean().item()}, avg cont: {continuation_values.mean().item()}")


                # Update continuation value
                with torch.no_grad():
                    logits = phi @ self.weights[t]
                    sigma = torch.sigmoid(logits)
                    payoffs = torch.tensor([self.payoff_fn(t, traj[t]) for traj in trajectories], dtype=torch.float32)
                    cont = continuation_values
                    mask = payoffs > 0
                    continuation_values[mask] = payoffs[mask] * sigma[mask] + cont[mask] * (1 - sigma[mask])

        #     print(f"Weights at iteration {iter_num + 1}:")
        #     for t in range(self.steps):
        #         print(f"t: {t}, weights: {self.weights[t].detach().numpy()}")

        # print("Final weights:")
        # for t in range(self.steps):
        #     print(f"t: {t}, weights: {self.weights[t].detach().numpy()}")

        if save_weights:
            torch.save(self.weights, save_path)
            print(f"Weights saved to {save_path}")

    def should_stop(self, t, x):
        """
        Decide whether to stop at (t, x) using the learned policy.

        Parameters:
        - t (int): Time step
        - x (float): Asset price

        returns (bool): Whether to stop
        """
        if t >= self.steps:
            return False
        phi_x = self._phi(x)
        score = torch.dot(self.weights[t], phi_x)
        return torch.sigmoid(score).item() > 0.5

    def get_exercise_boundary(self, t_values, x_grid, threshold=0.5):
        """
        Approximates the exercise boundary at each t by finding the x value where stop probability ≈ threshold.

        Parameters:
        - t_values (list): List of discrete time steps
        - x_grid (list): Range of x values to evaluate
        - threshold (float): Probability threshold for stopping

        returns (dict): Mapping of t to exercise boundary x*
        """
        boundaries = {}
        for t in t_values:
            if t >= self.steps:
                boundaries[t] = None
                continue
            for x in sorted(x_grid, key=lambda x: self.payoff_fn(t, x)):
                phi_x = self._phi(x)
                p_stop = torch.sigmoid(torch.dot(self.weights[t], phi_x)).item()
                # if abs(p_stop - threshold) < epsilon and self.payoff_fn(t, x) > 0:
                # if p_stop > threshold and self.payoff_fn(t, x) > 0:
                if p_stop >= threshold:
                    # print(f"t: {t}, x: {x}, p_stop: {p_stop}")
                    boundaries[t] = x
                    break
            else:
                boundaries[t] = None
                # print(f"No boundary found for t: {t}")
                # print(f"weights: {self.weights[t].detach().numpy()}")
                
        return boundaries
    

    def value(self, trajectory):
        """
        Compute reward earned following policy on a single trajectory.

        Parameters:
        - trajectory (list): Sequence of x values

        returns (float): Payoff collected
        """
        for t, x_t in enumerate(trajectory):
            if self.should_stop(t, x_t):
                return self.payoff_fn(t, x_t)
        return self.payoff_fn(self.steps, trajectory[-1])

    def get_stop_probability(self, t, x):
        """
        Return the model's stopping probability at time t and asset value x.

        Parameters:
        - t (int): Time step
        - x (float): Asset price

        Returns:
        - float: Probability of stopping
        """
        if t >= self.steps:
            return 0.0
        phi_x = self._phi(x)
        score = torch.dot(self.weights[t], phi_x)
        return torch.sigmoid(score).item()
    
    def get_weights(self):
        """
        Get the weights of the regression models for each time step.

        Returns:
        - weights (list of np.ndarray): List of weights for each time step
        """
        return [
            model.detach().numpy() if model is not None else None for model in self.weights
        ]
    