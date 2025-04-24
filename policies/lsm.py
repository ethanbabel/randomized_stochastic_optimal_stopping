import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
from policies.stopping_policy import StoppingPolicy

class LSMPolicy(StoppingPolicy):
    def __init__(self, basis_functions, payoff_fn, T):
        """
        LSM stopping policy using linear regression for continuation value.

        Parameters:
        - basis_functions (list of functions): Basis functions used for regression
        - payoff_fn (function): Payoff function g(t, x_t)
        - T (int): Total number of timesteps
        """
        super().__init__()
        self.basis_functions = basis_functions
        self.payoff_fn = payoff_fn
        self.T = T
        self.models = [None] * (T)  # regression models per timestep
        self.improvement_threshold = 0.01 # threshold above cont_val for exercise
        self.trained = False

    def _phi(self, x):
        """
        Evaluate basis functions on input x (usually the state of the stochastic process governing the system).
        Parameters:
        - x (float or np.ndarray): Input value(s) for basis functions
        Returns:
        - np.ndarray: Evaluated basis functions
        """
        return np.array([f(x) for f in self.basis_functions], dtype=np.float32)

    def train(self, trajectories):
        """
        Train LSM policy on given trajectories.

        Parameters:
        - trajectories (np.ndarray): A numpy array of shape (n_paths, T+1) representing the trajectories of the underlying process.
            Each row corresponds to a path, and each column corresponds to a time step.
        """
        n_paths, T_plus_1 = trajectories.shape
        if T_plus_1 != self.T + 1:
            raise ValueError(f"Expected trajectories of shape (n_paths, {self.T + 1}), got {trajectories.shape}")
        
        payoffs = np.zeros((n_paths, self.T))

        # Precompute payoffs
        for t in range(self.T):
            for i in range(n_paths):
                payoffs[i, t] = self.payoff_fn(t, trajectories[i, t])

        # Initialize cashflows to terminal payoffs
        cashflows = payoffs[:, -1].copy()

        for t in reversed(range(0, self.T)):
            X = []
            Y = []

            for i in range(n_paths):
                x_t = trajectories[i, t]
                if payoffs[i, t] >=0:  # only regress on in-the-money paths
                    X.append(self._phi(x_t))
                    Y.append(cashflows[i])
            
            if len(X) == 0:
                continue

            X = np.array(X)
            Y = np.array(Y)

            # model = LinearRegression(fit_intercept=False)
            model = Ridge(alpha=1.0, fit_intercept=False)
            model.fit(X, Y)
            if model is None:
                raise ValueError("Model training failed.")
            self.models[t] = model

            # Calculate continuation values and update cashflows
            for i in range(n_paths):
                x_t = trajectories[i, t]
                phi_x = np.array(self._phi(x_t)).reshape(1, -1)
                cont_val = model.predict(phi_x)[0]
                cont_val = max(0, cont_val)  # Ensure non-negative continuation value
                if payoffs[i, t] >= cont_val:
                    cashflows[i] = payoffs[i, t]

        self.trained = True

    def should_stop(self, t, x_t):
        """
        Determine whether to stop at time t given state x_t.

        Parameters:
        - t (int): Time step
        - x_t (float): State at time t
        Returns:
        - exercise decision (bool): True if stopping is optimal, False otherwise
        """
        if not self.trained or self.models[t] is None:
            return False
        phi_x = torch.tensor(self._phi(x_t)).unsqueeze(0)
        cont_val = self.models[t].predict(phi_x)[0]
        cont_val = max(0, cont_val)  # Ensure non-negative continuation value
        return self.payoff_fn(t, x_t) >= cont_val

    def get_exercise_boundary(self, t_values, x_grid):
        """
        Estimate the exercise boundary for each given time step.

        Parameters:
        - t_values (list of int): List of time steps at which to compute exercise boundaries
        - x_grid (list or np.ndarray): Grid of state values to evaluate stopping condition over

        Returns:
        - boundaries (dict): A dictionary mapping each time t to the smallest x in x_grid (ower bound of the stopping region) 
            such that stopping is optimal. If no such x exists, the value is None.
        """
        boundaries = {}
        for t in t_values:
            model = self.models[t]
            if model is None:
                boundaries[t] = None
                continue
            for x in sorted(x_grid, key=lambda x: self.payoff_fn(t, x)):
                phi_x = np.array(self._phi(x)).reshape(1, -1)
                cont_val = model.predict(phi_x)[0]
                cont_val = max(0, cont_val)  # Ensure non-negative continuation value
                if self.payoff_fn(t, x) > cont_val:
                    # print(f"t: {t}, x: {x}, cont_val: {cont_val}")
                    boundaries[t] = x
                    break
            else:
                boundaries[t] = None
        return boundaries

    def value(self, trajectory):
        """
        Estimate the value of the stopping rule for a single trajectory.

        Parameters:
        - trajectory (np.ndarray): A numpy array of shape (T+1,) representing a single trajectory of the underlying process.
        Returns:
        - float: Estimated value of the stopping rule on the given trajectory.
        """
        for t in range(self.T):
            if self.should_stop(t, trajectory[t]):
                return self.payoff_fn(t, trajectory[t])
        # If we never stop before T, collect terminal payoff
        return self.payoff_fn(self.T, trajectory[self.T])
    
    def get_weights(self):
        """
        Get the weights of the regression models for each time step.

        Returns:
        - weights (list of np.ndarray): List of weights for each time step
        """
        return [
            model.coef_ if model is not None else None for model in self.models
        ]
