from abc import ABC, abstractmethod

class StoppingPolicy(ABC):
    def __init__(self):
        self.trained = False

    @abstractmethod
    def train(self, trajectories):
        """Train or fit the stopping policy on sample trajectories."""
        pass

    @abstractmethod
    def should_stop(self, t, x_t):
        """
        Return True/False (or a probability) indicating whether to stop at time t
        given state x_t. Used for exercise boundary plotting and backtesting.
        """
        pass

    @abstractmethod
    def get_exercise_boundary(self, t_values, x_grid):
        """
        Return the boundary value(s) at each time t where the stopping decision changes.
        Used for visualizing exercise boundaries.
        """
        pass

    @abstractmethod
    def value(self, trajectory):
        """
        Estimate the value of the stopping rule on a given trajectory.
        """
        pass