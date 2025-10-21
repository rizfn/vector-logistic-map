import numpy as np
import matplotlib.pyplot as plt

def main():
    # Parameters matching JavaScript
    alpha_fixed = 10.0
    b_min = 0.5
    b_max = 1.5
    b_steps = 5000

    # Simulation parameters
    n_transient = 500  # Discard initial transient
    n_plot = 1000       # Number of points to plot per b value
    x0 = 0.5           # Initial condition

    # Gaussian map function
    def gauss_map(x, b, alpha=alpha_fixed):
        return b * np.exp(-alpha * (x - 0.5)**2)

    # Generate bifurcation diagram
    b_values = np.linspace(b_min, b_max, b_steps)
    x_bifurcation = []
    b_bifurcation = []

    for b in b_values:
        x = x0
        # Transient
        for _ in range(n_transient):
            x = gauss_map(x, b)
        # Collect points
        for _ in range(n_plot):
            x = gauss_map(x, b)
            x_bifurcation.append(x)
            b_bifurcation.append(b)

    # Plot
    plt.figure(figsize=(10, 6), facecolor='none', dpi=300)
    plt.plot(b_bifurcation, x_bifurcation, ',k', markersize=0.5, alpha=0.1)
    plt.xlabel('b', fontsize=20)
    plt.ylabel('x', fontsize=20)
    plt.title(r'Bifurcation diagram: $f(x) = b \, e^{-10(x-0.5)^2}$', fontsize=16)
    plt.xlim(b_min, b_max)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/visualizations/cmlTest2D/gauss_map_bifurcation.png', dpi=300, bbox_inches='tight')

    print("Bifurcation diagram saved to docs/visualizations/cmlTest2D/gauss_map_bifurcation.png")

if __name__ == "__main__":
    main()