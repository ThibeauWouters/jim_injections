import os
os.environ["CUDA_VISIBILE_DEVICES"] = ""
import psutil
p = psutil.Process()
p.cpu_affinity([0])
import copy
import numpy as np
import json
import arviz 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.stats import kstest, uniform, percentileofscore
import jax.numpy as jnp

from ripple import get_chi_eff, Mc_eta_to_ms, lambdas_to_lambda_tildes, lambda_tildes_to_lambdas
from jimgw.prior import Uniform, Composite
from jimgw.single_event.waveform import RippleTaylorF2
from jimgw.single_event.detector import H1, L1, V1
from astropy.time import Time

import sys
sys.path.append("../tidal/")
from injection_recovery import PRIOR, NAMING

from tqdm import tqdm

### Hyperparameters
matplotlib_params = {
    "font.size": 22,
    "legend.fontsize": 22,
    "legend.frameon": False,
    "axes.labelsize": 22,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.figsize": (7, 5),
    "xtick.top": True,
    "axes.unicode_minus": False,
    "ytick.right": True,
    "xtick.bottom": True,
    "ytick.left": True,
    "xtick.major.pad": 8,
    "xtick.major.size": 6,
    "xtick.minor.size": 2,
    "ytick.major.size": 6,
    "ytick.minor.size": 2,
    "axes.linewidth": 1.5,
    "text.usetex": True,
    # "font.family": "serif",
    # "font.serif": "cmr10",
    # "mathtext.fontset": "cm",
    # "axes.formatter.use_mathtext": True,  # needed when using cm=cmr10 for normal text
}
plt.rcParams.update(matplotlib_params)

default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=24),
                        title_kwargs=dict(fontsize=24), 
                        color="blue",
                        levels=[0.68, 0.95, 0.997],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False)

labels_tidal_deltalambda = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\tilde{\Lambda}$', r'$\delta\tilde{\Lambda}$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\cos\iota$', r'$\psi$', r'$\alpha$', r'$\sin\delta$']

labels_tidal_lambda12 = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda_1$', r'$\Lambda_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\cos\iota$', r'$\psi$', r'$\alpha$', r'$\sin\delta$']

labels_tidal_deltalambda_chi_eff = [r'$M_c/M_\odot$', r'$q$', r'$\chi_{\rm eff}$', r'$\tilde{\Lambda}$', r'$\delta\tilde{\Lambda}$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\cos\iota$', r'$\psi$', r'$\alpha$', r'$\sin\delta$']

labels_tidal_lambda12_chi_eff = [r'$M_c/M_\odot$', r'$q$', r'$\chi_{\rm eff}$', r'$\Lambda_1$', r'$\Lambda_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\cos\iota$', r'$\psi$', r'$\alpha$', r'$\sin\delta$']

labels_tidal = labels_tidal_lambda12 = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda_1$', r'$\Lambda_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\cos\iota$', r'$\psi$', r'$\alpha$', r'$\sin\delta$']

#########################
### PP-PLOT UTILITIES ###
#########################

def get_mirror_location(samples: np.array) -> tuple[np.array, np.array]:
    """Computes the mirrored location of the samples in the sky.

    Args:
        samples (np.array): Posterior values or true values

    Returns:
        tuple[np.array, np.array]: The mirrored samples in the sky, two versions for two signs of t_c.
    """
    
    # Just to be sure, make a deepcopy
    mirror_samples = copy.deepcopy(samples)
    
    # Check whether we have a list of samples, or a single sample
    if len(np.shape(mirror_samples)) == 1:
        mirror_samples = np.array([mirror_samples])
        
    # Get the parameter names
    naming = list(PRIOR.keys())
    
    # Get indices of parameters for which we will perform a transformation
    alpha_index = naming.index("ra")
    delta_index = naming.index("sin_dec")
    iota_index  = naming.index("cos_iota")
    phi_c_index = naming.index("phase_c")
    t_c_index   = naming.index("t_c")
    
    # First transform iota and delta
    mirror_samples[:, iota_index] = np.arccos(mirror_samples[:, iota_index])
    mirror_samples[:, delta_index] = np.arcsin(mirror_samples[:, delta_index])
    
    # Do the transformations:
    mirror_samples[:, alpha_index] = (mirror_samples[:, alpha_index] + np.pi) % (2 * np.pi)
    mirror_samples[:, delta_index] =  - mirror_samples[:, delta_index]
    mirror_samples[:, iota_index]  =  np.pi - mirror_samples[:, iota_index]
    mirror_samples[:, phi_c_index] = (mirror_samples[:, phi_c_index] + np.pi) % (2 * np.pi)
    
    ## TODO check the t_c transformation
    R_e = 6.378e+6
    c   = 299792458
    # Will have on with plus, and one with minus, so copy whatever we have already now
    second_mirror_samples = copy.deepcopy(mirror_samples)
    # Also will return a copy where t_c was not changed:
    mirror_samples_same_tc = copy.deepcopy(mirror_samples)
    mirror_samples[:, t_c_index] = mirror_samples[:, t_c_index] - R_e / c
    second_mirror_samples[:, t_c_index] = second_mirror_samples[:, t_c_index] + R_e / c
    
    # Convert iota and delta back to cos and sin values
    mirror_samples[:, iota_index] = np.cos(mirror_samples[:, iota_index])
    mirror_samples[:, delta_index] = np.sin(mirror_samples[:, delta_index])
    second_mirror_samples[:, iota_index] = np.cos(second_mirror_samples[:, iota_index])
    second_mirror_samples[:, delta_index] = np.sin(second_mirror_samples[:, delta_index])
    
    return mirror_samples_same_tc, mirror_samples, second_mirror_samples

def make_cumulative_histogram(data: np.array, nb_bins: int = 100):
    """
    Creates the cumulative histogram for a given dataset.

    Args:
        data (np.array): Given dataset to be used.
        nb_bins (int, optional): Number of bins for the histogram. Defaults to 100.

    Returns:
        np.array: The cumulative histogram, in density.
    """
    h = np.histogram(data, bins = nb_bins, range=(0,1), density=True)
    return np.cumsum(h[0]) / 100.0

def make_uniform_cumulative_histogram(size: tuple, nb_bins: int = 100) -> np.array:
    """
    Generates a cumulative histogram from uniform samples.
    
    Size: (N, dim): Number of samples from the uniform distribution, counts: nb of samples

    Args:
        counts (int): Dimensionality to be generated.
        N (int, optional): Number of samples from the uniform distribution. Defaults to 10000.

    Returns:
        np.array: Cumulative histograms for uniform distributions. Shape is (N, nb_bins)
    """
    
    uniform_data = np.random.uniform(size = size)
    cum_hist = []
    for data in uniform_data:
        result = make_cumulative_histogram(data, nb_bins = nb_bins)
        cum_hist.append(result)
        
    cum_hist = np.array(cum_hist)
    return cum_hist

def get_true_params_and_credible_level(chains: np.array, 
                                       true_params_list: np.array,
                                       return_first: bool = False) -> tuple[np.array, float]:
    """
    Finds the true parameter set from a list of true parameter sets, and also computes its credible level.

    Args:
        true_params_list (np.array): List of true parameters and copies for sky location.

    Returns:
        tuple[np.array, float]: The select true parameter set, and its credible level.
    """
    
    # Indices which have to be treated as circular
    circular_idx = [8, 9, 10, 11, 12]
    
    if return_first:
        # Ignore the sky location mirrors, just take the first one
        true_params = true_params_list[0]
        true_params_list = [true_params]
    
    # When checking sky reflected as well, iterate over all "copies"
    for i, true_params in enumerate(true_params_list):
        params_credible_level_list = []
        
        # ### OLD code
        # boolean_values = np.array(chains < true_params)
        # credible_level = np.mean(boolean_values, axis = 0)
        # params_credible_level_list = credible_level
        
        # Iterate over each parameter of this "copy" of parameters
        for j, param in enumerate(true_params):
            
            # if j in circular_idx:
            #     pass
            # else:
            q = percentileofscore(chains[:, j], param)
            q /= 100.0
            
            # Two sided version:
            credible_level = 1 - 2 * min(q, 1-q)
            # # One sided version:
            # credible_level = q
                
            params_credible_level_list.append(credible_level)
        
        params_credible_level_list = np.array(params_credible_level_list)
        
        if i == 0:
            credible_level_list = params_credible_level_list 
        else:
            credible_level_list = np.vstack((credible_level_list, params_credible_level_list))
            
        # Now choose the correct index
        if return_first:
            credible_level_list = np.reshape(credible_level_list, (1, -1))
        summed_credible_level = np.sum(abs(0.5 - credible_level_list), axis = 1)
        # Pick the index with the lowest sum
        idx = np.argmin(summed_credible_level)
        true_params = true_params_list[idx]
        credible_level = credible_level_list[idx]
    
    return true_params, credible_level

def get_credible_levels_injections(outdir: str, 
                                   reweigh_distance: bool = False,
                                   return_first: bool = True,
                                   max_number_injections: int = -1) -> np.array:
    """
    Compute the credible levels list for all the injections. 
    
    Args:
        reweigh_distance (bool, optional): Whether to reweigh based on the distance or not. Defaults to False.
        return_first (bool, optional): Whether to return the first true parameter set (don't take sky location mirrors into account) or not. Defaults to False.

    Returns:
        np.array: Array of credible levels for each injection.
    """
    
    # Get parameter names
    naming = list(PRIOR.keys())
    print("naming")
    print(naming)
    n_dim = len(naming)
    
    print("Reading injection results")
    
    credible_level_list = []
    subdirs = []
    counter = 0
    
    print("Iterating over the injections, going to compute the credible levels")
    for subdir in tqdm(os.listdir(outdir)):
        subdir_path = os.path.join(outdir, subdir)
        
        if os.path.isdir(subdir_path):
            json_path = os.path.join(subdir_path, "config.json")
            chains_filename = f"{subdir_path}/results_production.npz"
            if not os.path.isfile(json_path) or not os.path.isfile(chains_filename):
                continue
            
            subdirs.append(subdir)
            
            # Load config, and get the injected parameters
            with open(json_path, "r") as f:
                config = json.load(f)
            true_params = np.array([config[name] for name in naming])
                
            # Get the recovered parameters
            data = np.load(chains_filename)
            chains = data['chains'].reshape(-1, n_dim)
            
            # Reweigh distance if needed
            if reweigh_distance:
                print("INFO: Reweighing distance for credible levels injections. . .")
                d_L_index = naming.index("d_L")
                d_values = chains[:, d_L_index]
                weights = d_values ** 2
                # Normalize the weights
                weights /= np.sum(weights)
                # Resample the chains based on these weights
                indices = np.random.choice(np.arange(len(chains)), size=len(chains), p=weights)
                chains = chains[indices]
                print("INFO: Resampled chains!")
            
            # # Get the sky mirrored values as well, NOTE this outputs an array of arrays!
            mirrored_values = get_mirror_location(true_params) # tuple
            mirrored_values = np.vstack(mirrored_values) # np array
            all_true_params = np.vstack((true_params, mirrored_values))
            
            true_params, credible_level = get_true_params_and_credible_level(chains, all_true_params, return_first=return_first)
            
            credible_level_list.append(credible_level)
            
            counter += 1
            if counter == max_number_injections:
                print(f"INFO: Stopping after {max_number_injections} injections.")
                break
            
    credible_level_list = np.array(credible_level_list)
    
    return credible_level_list, subdirs

def compute_chi_eff(mc, q, chi1, chi2):
    eta = q/(1+q)**2
    m1, m2 = Mc_eta_to_ms(jnp.array([mc, eta]))
    params = jnp.array([m1, m2, chi1, chi2])
    chi_eff = get_chi_eff(params)
    
    return chi_eff

#################
### DEBUGGING ###
#################

def plot_distributions_injections(outdir: str, 
                                  param_index: int = 0,
                                  **plotkwargs) -> None:
    """
    TODO
    
    By default, we are checking chirp mass
    """
    naming = list(PRIOR.keys())
    param_name = naming[param_index]
    print("Checking parameter: ", param_name)
    
    plt.figure(figsize=(12, 9))
    print("Iterating over the injections, going to compute the credible levels")
    for subdir in tqdm(os.listdir(outdir)):
        subdir_path = os.path.join(outdir, subdir)
        
        if os.path.isdir(subdir_path):
            json_path = os.path.join(subdir_path, "config.json")
            chains_filename = f"{subdir_path}/results_production.npz"
            if not os.path.isfile(json_path) or not os.path.isfile(chains_filename):
                continue
            
            # Load config, and get the true (injected) parameters
            with open(json_path, "r") as f:
                config = json.load(f)
            true_params = np.array([config[name] for name in naming])
            
            true_param = true_params[param_index]
            
            # Get distribution of samples
            data = np.load(chains_filename)
            chains = data['chains'].reshape(-1, len(naming))
            samples = chains[:, param_index]
            
            samples -= true_param
            # Make histogram
            plt.hist(samples, **plotkwargs)
            
    plt.axvline(0, color = "black", linewidth = 1)
    plt.xlabel(f"Residuals of {param_name}")
    plt.ylabel("Density")
    plt.savefig(f"./pp_TaylorF2/distributions_{param_name}.png")
    plt.close()
            
            
def plot_credible_levels_injections(outdir: str,
                                    param_index: int = 0
) -> None:
    """
    TODO
    """
    naming = list(PRIOR.keys())
    param_name = naming[param_index]
    print("Checking parameter: ", param_name)
    
    credible_level_list, _ = get_credible_levels_injections(outdir)
    
    plt.figure(figsize=(12, 9))
    plt.hist(credible_level_list[:, param_index], bins = 20, histtype="step", color="blue", linewidth=2, density=True)
    plt.xlabel("Credible level")
    plt.ylabel("Density")
    plt.title(f"Credible levels {param_name}")
    plt.savefig(f"./pp_TaylorF2/credible_levels_{param_name}.png")
    plt.close()


def analyze_credible_levels(credible_levels, 
                            subdirs, 
                            param_index = 0, 
                            nb_round: int = 5):
    subdirs = np.array(subdirs)
    
    credible_levels_param = credible_levels[:, param_index]
    
    # Sort 
    sorted_indices = np.argsort(credible_levels_param)
    credible_levels_param = credible_levels_param[sorted_indices]
    credible_levels = credible_levels[sorted_indices]
    subdirs_sorted = subdirs[sorted_indices]
    
    for i, (subdir, credible_level) in enumerate(zip(subdirs_sorted, credible_levels_param)):
        print(f"{subdir}: {np.round(credible_level, nb_round)}")

####################################
### Further postprocessing tools ###
####################################

def test_tile():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    print("np.shape(array)")
    print(np.shape(array))

    # Repeat along the second axis to create the desired shape (1000, 5, 13)
    result_array = np.tile(array[:, np.newaxis, :], (1, 5, 1))

    print("np.shape(result_array)")
    print(np.shape(result_array))

    print(result_array[:, 0, :])
    
def my_format(number: float):
    return "{:.2f}".format(number)
    
def analyze_runtimes(outdir, verbose: bool = True):
    runtimes = []
    for dir in os.listdir(outdir):
        runtime_file = outdir + dir + "/runtime.txt"
        if not os.path.exists(runtime_file):
            continue
        runtime = np.loadtxt(runtime_file)
        runtimes.append(runtime)
        
    runtimes = np.array(runtimes)
    if verbose:
        print(f"Mean +- std runtime: {my_format(np.mean(runtimes))} +- {my_format(np.std(runtimes))} seconds")
        print(f"Min runtime: {my_format(np.min(runtimes))} seconds")
        print(f"Max runtime: {my_format(np.max(runtimes))} seconds")
        print(f"Median runtime: {my_format(np.median(runtimes))} seconds")

        print("\n\n")
        runtimes /= 60
        print(f"Mean +- std runtime: {my_format(np.mean(runtimes))} +- {my_format(np.std(runtimes))} minutes")
        print(f"Min runtime: {my_format(np.min(runtimes))} minutes")
        print(f"Max runtime: {my_format(np.max(runtimes))} minutes")
        print(f"Median runtime: {my_format(np.median(runtimes))} minutes")

        print("\n\n")
        print(f"Number of runs: {len(runtimes)}")
    
    return runtimes