import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Parameters (set to match your simulation, or parse from command line)
    N = 256
    a = 3.9
    epsilon = 0.01

    # Build filename
    fname = f"N_{N}_a_{a}_e_{epsilon}"

    # Find the output file relative to this script
    exe_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    tsv_path = os.path.join(exe_dir, "outputs/timeseries", fname + ".tsv")

    # Load data
    data = np.loadtxt(tsv_path, delimiter="\t")

    # Select only the final 100 steps
    data_final = data[-100:, :]
    num_sites = data_final.shape[1]
    num_steps = data_final.shape[0]
    cmap = plt.get_cmap('plasma')
    colors = [cmap(i / num_sites) for i in range(num_sites)]

    plt.figure(figsize=(10, 6))
    for i in range(num_sites):
        plt.plot(np.arange(num_steps), data_final[:, i], color=colors[i], linewidth=0.7)

    plt.xlabel("Time (step, final 100)")
    plt.ylabel("x value")
    plt.title("Coupled Map Lattice Trajectories (Final 100 Steps)")
    plt.tight_layout()
    plot_dir = os.path.join(exe_dir, "plots/timeseries_trajectories")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"{fname}_final100.svg"))

if __name__ == "__main__":
    main()