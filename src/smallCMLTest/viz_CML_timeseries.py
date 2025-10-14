import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Parameters (set to match your simulation, or parse from command line)
    N = 256
    a = 3.9
    epsilon = 0.1

    # Build filename
    fname = f"N_{N}_a_{a}_e_{epsilon}.tsv"

    # Find the output file relative to this script
    exe_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    tsv_path = os.path.join(exe_dir, "outputs/timeseries", fname)

    # Load data
    data = np.loadtxt(tsv_path, delimiter="\t")

    plt.figure(figsize=(10, 6))
    plt.imshow(data, aspect='auto', cmap='plasma', interpolation='nearest')
    plt.xlabel("Space (site index)")
    plt.ylabel("Time (step)")
    plt.title("Coupled Map Lattice Time Evolution")
    plt.colorbar(label="x value")
    plt.tight_layout()
    plot_dir = os.path.join(exe_dir, "plots/timeseries")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"{fname}.png"))

if __name__ == "__main__":
    main()