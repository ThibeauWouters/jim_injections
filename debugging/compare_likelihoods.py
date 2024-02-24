import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"
import psutil
p = psutil.Process()
p.cpu_affinity([0])

from tqdm import tqdm
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
USE_VMAP = True

N_samples = 50

my_dict = {}

for WHICH_LIKELIHOOD in ["ML", "original"]:
    
    my_dict[WHICH_LIKELIHOOD] = {}
    
    print("====")
    print(f"WHICH_LIKELIHOOD = {WHICH_LIKELIHOOD}")
    print("====")

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
        cos_iota_prior,
        psi_prior,
        ra_prior, 
        sin_dec_prior,
    ]
    complete_prior = Composite(prior_list)
    bounds = jnp.array([[p.xmin, p.xmax] for p in complete_prior.priors])
    print("Finished prior setup")

    # Load the likelihood
    if WHICH_LIKELIHOOD == "ML":
        print("Loading ML likelihood")
        ext = "_ML"
    else:
        print("Loading original likelihood")
        ext = ""

    likelihood_filename = f'{OUTDIR}likelihood{ext}.pkl'
    print(likelihood_filename)

    with open(likelihood_filename, 'rb') as f:
        likelihood = pickle.load(f)

    # Load config.json
    with open(f'{OUTDIR}config.json', 'r') as f:
        config = json.load(f)
        
    true_params = jnp.array([config[p] for p in naming])
    true_params_np = np.array([config[p] for p in naming])
    true_mc = true_params_np[0]
    print("true_mc")
    print(true_mc)
    true_params_named = complete_prior.add_name(true_params)
    true_params_named["iota"] = jnp.arccos(true_params_named["cos_iota"])
    true_params_named["dec"] = jnp.arcsin(true_params_named["sin_dec"])
    q = true_params_named["q"]
    true_params_named["eta"] = q / (1 + q)**2

    # Get the max likelihood parameters
    name = f"{OUTDIR}results_production.npz"
    data = np.load(name)
    log_prob = data["log_prob"]
    log_prob = jnp.reshape(log_prob, (-1,))
    chains = data["chains"]
    chains = jnp.reshape(chains, (-1, 13))
    # Remove the prior prob
    prior_params = complete_prior.add_name(chains.T)
    prior_prob = complete_prior.log_prob(prior_params)
    log_prob -= prior_prob

    # Find max likelihood params
    max_likelihood_idx = jnp.argmax(log_prob)
    max_likelihood_params = chains[max_likelihood_idx]
    max_likelihood_params = np.array(max_likelihood_params)
    print("max_likelihood_params")
    print(max_likelihood_params)

    # Now go and generate the samples:
    if WHICH_LIKELIHOOD == "ML":
        my_params_stacked = np.vstack(N_samples * [max_likelihood_params])
        center = max_likelihood_params[0]
        mc_std = 0.001
    else:
        my_params_stacked = np.vstack(N_samples * [true_params_np])
        center = true_params_np[0]
        mc_std = 0.001

    mc_samples = np.linspace(center - mc_std, center + mc_std, N_samples)

    my_params_stacked[:, 0] = mc_samples
    my_params_stacked = jnp.array(my_params_stacked).T
    my_params_stacked_named = complete_prior.add_name(my_params_stacked)

    # Convert cos_iota to iota, sin_dec to dec, and q to eta
    my_params_stacked_named["iota"] = jnp.arccos(my_params_stacked_named["cos_iota"])
    my_params_stacked_named["dec"] = jnp.arcsin(my_params_stacked_named["sin_dec"])
    q = my_params_stacked_named["q"]
    my_params_stacked_named["eta"] = q / (1 + q)**2

    # Evaluate the samples
    print("Making vmap")
    likelihood_evaluate_vmap = jax.vmap(likelihood.evaluate, in_axes=(0, None))
    likelihood_evaluate_original_vmap = jax.vmap(likelihood.evaluate_original, in_axes=(0, None))
    print("Making vmap DONE")

    print("Computing")
    if USE_VMAP:
        log_likelihoods = likelihood_evaluate_vmap(my_params_stacked_named, {})
        log_likelihoods_original = likelihood_evaluate_original_vmap(my_params_stacked_named, {})
    else:
        log_likelihoods = []
        log_likelihoods_original = []
        
        jnp.shape(my_params_stacked)
        
        for param in tqdm(my_params_stacked.T):
            param_named = complete_prior.add_name(param)
            
            param_named['iota'] = jnp.arccos(param_named['cos_iota'])
            param_named['dec'] = jnp.arcsin(param_named['sin_dec'])
            q = param_named['q']
            param_named['eta'] = q / (1 + q)**2
            
            log_likelihoods.append(likelihood.evaluate(param_named, {}))
            log_likelihoods_original.append(likelihood.evaluate_original(param_named, {}))
        
        log_likelihoods = np.array(log_likelihoods)
        log_likelihoods_original = np.array(log_likelihoods_original)
        
    my_dict[WHICH_LIKELIHOOD] = {"log_likelihoods": log_likelihoods,
                                 "log_likelihoods_original": log_likelihoods_original}
    
    print("Computing DONE")

    print("Also computing it at the max L params")
    max_likelihood_params_named = complete_prior.add_name(max_likelihood_params)
    max_likelihood_params_named['iota'] = jnp.arccos(max_likelihood_params_named['cos_iota'])
    max_likelihood_params_named['dec'] = jnp.arcsin(max_likelihood_params_named['sin_dec'])
    q = max_likelihood_params_named['q']
    max_likelihood_params_named['eta'] = q / (1 + q)**2

    log_L_rb_max_L_params = likelihood.evaluate(max_likelihood_params_named, {})
    log_L_max_L_params = likelihood.evaluate_original(max_likelihood_params_named, {})

    print("Difference at max L params:")
    diff = (log_L_rb_max_L_params - log_L_max_L_params) / log_L_max_L_params
    print(diff)

    print("Difference at the true params:")
    log_L_rb_true_params = likelihood.evaluate(true_params_named, {})
    log_L_true_params = likelihood.evaluate_original(true_params_named, {})
    diff = (log_L_rb_true_params - log_L_true_params) / log_L_true_params
    print(diff)

    # Load the production samples
    name = f"{OUTDIR}results_production.npz"
    data = np.load(name)
    chains = data["chains"]
    chains = jnp.reshape(chains, (-1, 13))
    production_mc = chains[:, 0]
    producion_mc_median = np.median(production_mc)
    producion_mc_mean = np.mean(production_mc)

    print("Saving results")
    np.savez("results.npz", my_params_stacked=my_params_stacked, log_likelihoods=log_likelihoods, log_likelihoods_original=log_likelihoods_original)

    print("Plotting them")
    max_L_mc = 1.95713842
    fig = plt.figure(figsize=(10, 5))
    plt.plot(mc_samples, log_likelihoods, label="Relative binning")
    plt.plot(mc_samples, log_likelihoods_original, label="Original")
    plt.axvline(center, color = "black", linestyle = "--", label="Reference")
    plt.axvline(true_mc, color = "red", linestyle = "--", label="Injected value")
    plt.axvline(producion_mc_mean, color = "green", linestyle = "--", label="Production mean")
    plt.xlabel(r"$M_c \ [M_\odot]$")
    plt.ylabel("log likelihood")
    plt.legend()
    plt.savefig(f"./figures/likelihood_comparison{ext}.png")
    plt.show()

    print("Plotting relative difference")
    fig = plt.figure(figsize=(10, 5))
    rel_diff = (log_likelihoods - log_likelihoods_original) / log_likelihoods_original
    plt.plot(mc_samples, rel_diff, color = "blue")
    plt.axvline(center, color = "black", linestyle = "--", label="Reference")
    plt.axvline(true_mc, color = "red", linestyle = "--", label="Injected value")
    plt.axvline(producion_mc_mean, color = "green", linestyle = "--", label="Production mean")
    plt.xlabel(r"$M_c \ [M_\odot]$")
    plt.ylabel(r"$(\ln \mathcal{L}_{\rm RB} - \ln \mathcal{L}) / \ln \mathcal{L}$")
    plt.legend()
    plt.savefig(f"./figures/likelihood_comparison_rel_diff{ext}.png")
    plt.show()

    print("Plotting relative difference magnitude")
    fig = plt.figure(figsize=(10, 5))
    plt.plot(mc_samples, abs(rel_diff), color = "blue")
    plt.axvline(center, color = "black", linestyle = "--", label="Reference")
    plt.axvline(true_mc, color = "red", linestyle = "--", label="Injected value")
    plt.axvline(producion_mc_mean, color = "green", linestyle = "--", label="Production mean")
    plt.xlabel(r"$M_c \ [M_\odot]$")
    plt.ylabel(r"$\left|(\ln \mathcal{L}_{\rm RB} - \ln \mathcal{L}) / \ln \mathcal{L} \right|$")
    plt.legend()
    plt.yscale("log")
    plt.savefig(f"./figures/likelihood_comparison_rel_diff_abs{ext}.png")
    plt.show()

    my_dict[WHICH_LIKELIHOOD]["mc_samples"] = mc_samples    

# Plot the comparison

print("Plotting total comparison")

fig = plt.figure(figsize=(10, 5))
plt.plot(my_dict["ML"]["mc_samples"], my_dict["ML"]["log_likelihoods"], label="Relative binning (ML)")
plt.plot(my_dict["original"]["mc_samples"], my_dict["original"]["log_likelihoods"], label="Relative binning (injection)")

plt.plot(my_dict["ML"]["mc_samples"], my_dict["ML"]["log_likelihoods_original"], label="Original (ML)")
plt.plot(my_dict["original"]["mc_samples"], my_dict["original"]["log_likelihoods_original"], label="Original (injection)")
plt.xlabel(r"$M_c \ [M_\odot]$")
plt.ylabel(r"$\ln \mathcal{L}$")
plt.legend()
plt.savefig(f"./figures/total_comparison_likelihoods.png")
plt.close()