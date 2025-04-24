import matplotlib.pyplot as plt

def plot_boundaries(policies, t_values, x_grid, title="Exercise Boundaries Comparison", filename="exercise_boundaries.png"):
    """
    Plot exercise boundaries from one or more stopping policies.

    Parameters:
    - policies (list of tuples): Each tuple is (label (str), policy (StoppingPolicy))
    - t_values (list of int): List of time steps to evaluate boundaries at
    - x_grid (list or np.ndarray): Grid of x values to sweep over when finding boundary
    - title (str): Title of the plot
    - filename (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    for label, policy in policies:
        boundary_dict = policy.get_exercise_boundary(t_values, x_grid)
        times = list(boundary_dict.keys())
        boundary_vals = [boundary_dict[t] for t in times]
        plt.plot(times, boundary_vals, label=label, marker='o')

    plt.xlabel("Time Step")
    plt.ylabel("Exercise Boundary")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# New function: plot_rpo_boundaries
def plot_rpo_boundaries(policies, t_values, x_grid, target_probs=[0.5], title="RPO Exercise Boundaries", filename="rpo_boundaries.png"):
    """
    Plot RPO exercise boundaries for multiple probability thresholds.

    Parameters:
    - policies (list of tuples): Each tuple is (label (str), policy (RPOPolicy))
    - t_values (list of int): List of time steps to evaluate boundaries at
    - x_grid (list or np.ndarray): Grid of x values to sweep over when finding boundary
    - target_probs (list of float): Probabilities to draw exercise boundaries at
    - title (str): Title of the plot
    - filename (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    for label, policy in policies:
        for p in target_probs:
            boundary_dict = policy.get_exercise_boundary(t_values, x_grid, threshold=p)
            times = list(boundary_dict.keys())
            boundary_vals = [boundary_dict[t] for t in times]
            plt.plot(times, boundary_vals, label=f"{label} (p={p})", marker='o')

    plt.xlabel("Time Step")
    plt.ylabel("Exercise Boundary")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

