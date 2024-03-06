"""
Simple script to analyze the relative binning approximation and whether it is valid. 

This is only loading in a likelihood, and computing the likelihoods (both true and RB) for a set of samples.
"""

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"
import numpy as np 
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax 
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.single_event.waveform import RippleTaylorF2, RippleIMRPhenomD_NRTidalv2
from jimgw.prior import Uniform, Composite
import pickle
import time
from astropy.time import Time
from tqdm import tqdm
jax.config.update("jax_enable_x64", True)
print(jax.devices())

import utils

trigger_time = 1187008882.43
gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad
N_samples = 1_00

# Sample uniformly in prior range:
samples = np.random.uniform(utils.prior_low, utils.prior_high, size=(N_samples, len(utils.prior_low)))

print("samples")
print(samples)

# Load the likelihood
which_outdir = "outdir_NRTv2_binsize_1000"
name = "injection_8"
filename = f'../../tidal/{which_outdir}/{name}/likelihood.pickle'
with open(filename, 'rb') as f:
    likelihood = pickle.load(f)
    
print("likelihood")
print(likelihood)

### Evaluate likelihoods ###

### NEW VERSION WITH VMAP

start = time.time()
print("start evaluate")
log_likelihood = utils.my_evaluate_vmap(likelihood, utils.complete_prior, samples, original = False)
log_likelihood = np.array(log_likelihood)
print("end evaluate")
end = time.time()
print(f"Time taken: {end - start}")

print(log_likelihood)

start = time.time()
print("start evaluate original")
log_likelihood_original = utils.my_evaluate_vmap(likelihood, utils.complete_prior, samples, original = True)
log_likelihood_original = np.array(log_likelihood_original)
print("end evaluate original")
end = time.time()
print(f"Time taken: {end - start}")

diffs = abs((log_likelihood - log_likelihood_original) / log_likelihood_original)

print("diffs")
print(diffs)

print("Saving likelihoods")
np.savez(f"evaluated_likelihoods/{name}.npz", log_likelihood = log_likelihood, log_likelihood_original = log_likelihood_original, samples = samples)