import numpy as np
import matplotlib.pyplot as plt

def generate_gbm_trajectories(S0, mu, sigma, T, steps, n, seed=None):
    """
    Generate n geometric Brownian motion trajectories.

    Parameters:
    - S0 (float): Initial asset price
    - mu (float): Drift coefficient
    - sigma (float): Volatility coefficient
    - T (float): Total time horizon
    - steps (int): Number of discrete time steps
    - n (int): Number of trajectories to generate
    - seed (int, optional): Random seed for reproducibility (default: None)

    Returns:
    - trajectories (np.ndarray): A numpy array of shape (n, steps + 1) representing the trajectories of GBM
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps
    time_grid = np.linspace(0, T, steps + 1)

    # Brownian increments and paths
    dW = np.random.normal(0, np.sqrt(dt), size=(n, steps))
    W = np.cumsum(dW, axis=1)
    W = np.hstack((np.zeros((n, 1)), W))  # Include W(0) = 0

    # GBM formula: S(t) = S0 * exp((mu - 0.5 * sigma^2) * t + sigma * W(t))
    exponent = (mu - 0.5 * sigma ** 2) * time_grid + sigma * W
    S = S0 * np.exp(exponent)

    return S

def graph_gbm_trajectories(trajectories, T, steps, filename="section51_gbm_comparison/gbm_trajectories.png"):
    """
    Plot and save a graph of GBM trajectories.
    
    Parameters:
    - trajectories (np.ndarray): A numpy array of shape (n, steps + 1) representing GBM paths
    - T (float): Total time horizon
    - steps (int): Number of time steps
    - filename (str): File path to save the plot
    """

    time_grid = np.linspace(0, T, steps + 1)

    plt.figure(figsize=(10, 6))
    for i in range(trajectories.shape[0]):
        plt.plot(time_grid, trajectories[i], linewidth=0.8)

    plt.title("Geometric Brownian Motion Trajectories")
    plt.xlabel("Time")
    plt.ylabel("Asset Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
 
if __name__ == "__main__":
    # EXAMPLE USAGE:
    # Parameters for GBM
    S0 = 100          # Initial stock price
    mu = 0.05         # Drift
    sigma = 0.2       # Volatility
    T = 1.0           # Time horizon (1 year)
    steps = 252       # Daily steps for 1 year
    n = 10            # Number of trajectories
 
    # Generate and plot trajectories
    trajectories = generate_gbm_trajectories(S0, mu, sigma, T, steps, n)
    graph_gbm_trajectories(trajectories, T, steps)