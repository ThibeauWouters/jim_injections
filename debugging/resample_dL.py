import os
import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

outdir = "../tidal/outdir_TaylorF2_part2/"

params = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(params)

# Iterate over the directories

dL_values = []
for subdir in os.listdir(outdir):
    if os.path.isdir(outdir + subdir):
        path = outdir + subdir + "/config.json"
        with open(path) as f:
            config = json.load(f)
        dL_values.append(config["d_L"])
        
dL_values = np.array(dL_values)
print(dL_values)

# Get samples from a uniform range to compare
dL_min = 30.0
dL_max = 300.0

dL_uniform = np.random.uniform(dL_min, dL_max, 1_000)

# Get KDEs
x = np.linspace(dL_min, dL_max, 1_000)
kde = gaussian_kde(dL_values)
kde_uniform = lambda x: 1/(dL_max - dL_min) * np.ones_like(x)

# Plot the histogram of the dL values
lw = 2
plt.hist(dL_values, bins=20, histtype='step', density=True, label="Samples", linewidth=lw)
plt.hist(dL_uniform, bins=20, histtype='step', density=True, label="Uniform", linewidth=lw)
# Plot the KDE
plt.plot(x, kde(x), label="KDE", linewidth=lw)
plt.plot(x, kde_uniform(x), label="KDE Uniform", linewidth=lw)
plt.xlabel("dL")
plt.ylabel("Density")
plt.legend()
plt.savefig("./figures/dL_histogram.png")
plt.show()

# Save the KDE object
kde_file = "./kde_dL.npz"
np.savez(kde_file, x=x, y=kde(x))

# Test loading the data
data = np.load(kde_file)
print("data[x]")
print(data["x"])

print("data[y]")
print(data["y"])