import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"
import psutil
p = psutil.Process()
p.cpu_affinity([0])

import json
import numpy as np
import jax
import jax.numpy as jnp
from jimgw.prior import Uniform, Composite
import matplotlib.pyplot as plt
import pickle

from save_likelihoods import NAMING as naming 
from save_likelihoods import PRIOR

prior_ranges = jnp.array([PRIOR[name] for name in naming])
prior_low, prior_high = prior_ranges[:, 0], prior_ranges[:, 1]

OUTDIR = './outdir/injection_14/'

# Set up prior
print("Start prior setup")

# Priors without transformation
Mc_prior = Uniform(prior_low[0], prior_high[0], naming=['M_c'])
q_prior = Uniform(prior_low[1], prior_high[1], naming=['q'],
                    transforms={
    'q': (
        'eta',
        lambda params: params['q'] /
        (1 + params['q']) ** 2
    )
}
)
s1z_prior = Uniform(prior_low[2], prior_high[2], naming=['s1_z'])
s2z_prior = Uniform(prior_low[3], prior_high[3], naming=['s2_z'])
lambda_1_prior = Uniform(prior_low[4], prior_high[4], naming=['lambda_1'])
lambda_2_prior = Uniform(prior_low[5], prior_high[5], naming=['lambda_2'])
dL_prior = Uniform(prior_low[6], prior_high[6], naming=['d_L'])
tc_prior = Uniform(prior_low[7], prior_high[7], naming=['t_c'])
phic_prior = Uniform(prior_low[8], prior_high[8], naming=['phase_c'])
cos_iota_prior = Uniform(prior_low[9], prior_high[9], naming=["cos_iota"],
                            transforms={
    "cos_iota": (
        "iota",
        lambda params: jnp.arccos(
            jnp.arcsin(
                jnp.sin(params["cos_iota"] / 2 * jnp.pi)) * 2 / jnp.pi
        ),
    )
},
)
psi_prior = Uniform(prior_low[10], prior_high[10], naming=["psi"])
ra_prior = Uniform(prior_low[11], prior_high[11], naming=["ra"])
sin_dec_prior = Uniform(prior_low[12], prior_high[12], naming=["sin_dec"],
                        transforms={
    "sin_dec": (
        "dec",
        lambda params: jnp.arcsin(
            jnp.arcsin(
                jnp.sin(params["sin_dec"] / 2 * jnp.pi)) * 2 / jnp.pi
        ),
    )
},
)

# Compose the prior
prior_list = [
    Mc_prior,
    q_prior,
    s1z_prior,
    s2z_prior,
    lambda_1_prior,
    lambda_2_prior,
    dL_prior,
    tc_prior,
    phic_prior,
    cos_iota_prior, # 9
    psi_prior, # 10
    ra_prior, # 11 
    sin_dec_prior, # 12
]
complete_prior = Composite(prior_list)
bounds = jnp.array([[p.xmin, p.xmax] for p in complete_prior.priors])
print("Finished prior setup")

# Load the likelihood
with open(f'{OUTDIR}likelihood.pkl', 'rb') as f:
    likelihood = pickle.load(f)
print(likelihood)

# Load config.json
with open(f'{OUTDIR}config.json', 'r') as f:
    config = json.load(f)
    
true_params = jnp.array([config[p] for p in naming])
true_params_np = np.array([config[p] for p in naming])

# Now go and generate the samples:
N_samples = 100
true_params_stacked = np.vstack(N_samples * [true_params_np])
true_mc = true_params[0]
mc_std = 0.01
mc_samples = np.linspace(true_mc - mc_std, true_mc + mc_std, N_samples)

true_params_stacked[:, 0] = mc_samples
true_params_stacked = jnp.array(true_params_stacked).T
true_params_stacked_named = complete_prior.add_name(true_params_stacked)

# Convert cos_iota to iota, sin_dec to dec, and q to eta
true_params_stacked_named["iota"] = jnp.arccos(true_params_stacked_named["cos_iota"])
true_params_stacked_named["dec"] = jnp.arcsin(true_params_stacked_named["sin_dec"])
q = true_params_stacked_named["q"]
true_params_stacked_named["eta"] = q / (1 + q)**2

print(jnp.shape(true_params_stacked_named))

# Evaluate the samples
print("Making vmap")
likelihood_evaluate_vmap = jax.vmap(likelihood.evaluate, in_axes=(0, None))
likelihood_evaluate_original_vmap = jax.vmap(likelihood.evaluate_original, in_axes=(0, None))
print("Making vmap DONE")

print("Computing")
log_likelihoods = likelihood_evaluate_vmap(true_params_stacked_named, {})
log_likelihoods_original = likelihood_evaluate_original_vmap(true_params_stacked_named, {})
print("Computing DONE")

print(log_likelihoods)
print(log_likelihoods_original)

print("Saving results")
np.savez("results.npz", true_params_stacked=true_params_stacked, log_likelihoods=log_likelihoods, log_likelihoods_original=log_likelihoods_original)

print("Plotting them")
fig = plt.figure(figsize=(10, 5))
plt.plot(mc_samples, log_likelihoods, label="Relative binning")
plt.plot(mc_samples, log_likelihoods_original, label="Original")
plt.axvline(true_mc, color = "black", linestyle = "--", label="True value")
plt.xlabel(r"$M_c \ [M_\odot]$")
plt.ylabel("log likelihood")
plt.legend()
plt.savefig("likelihood_comparison.png")
plt.show()

print("Plotting relative difference")
fig = plt.figure(figsize=(10, 5))
rel_diff = (log_likelihoods - log_likelihoods_original) / log_likelihoods_original
plt.plot(mc_samples, rel_diff, color = "blue")
plt.axvline(true_mc, color = "black", linestyle = "--", label="True value")
plt.xlabel(r"$M_c \ [M_\odot]$")
plt.ylabel(r"$(\mathcal{L}_{\rm RB} - \mathcal{L}) / \mathcal{L}$")
plt.legend()
plt.savefig("likelihood_comparison_rel_diff.png")
plt.show()

print("Plotting relative difference magnitude")
fig = plt.figure(figsize=(10, 5))
plt.plot(mc_samples, abs(rel_diff), color = "blue")
plt.axvline(true_mc, color = "black", linestyle = "--", label="True value")
plt.xlabel(r"$M_c \ [M_\odot]$")
plt.ylabel(r"$\left|(\mathcal{L}_{\rm RB} - \mathcal{L}) / \mathcal{L} \right|$")
plt.legend()
plt.savefig("likelihood_comparison_rel_diff_abs.png")
plt.show()


# # Make histogram with numpy and plot
# fig = plt.figure(figsize=(10, 5))
# n_bins = 50
# hist, bin_edges = np.histogram(log_likelihoods, bins=n_bins)
# hist_rb, _ = np.histogram(log_likelihoods_rb, bins=bin_edges)

# plt.stairs(hist, bin_edges, histtype = "step", label="Original")
# plt.stairs(hist_rb, bin_edges, histtype = "step", density = True, label="Relative binning")

# plt.legend()
# plt.xlabel("log likelihood")
# plt.show()