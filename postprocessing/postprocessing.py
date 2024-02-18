import copy
import re
import pandas as pd
import numpy as np
import os
import json
import arviz 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.stats import kstest, uniform
import jax
import jax.numpy as jnp

from ripple import get_chi_eff, Mc_eta_to_ms, lambdas_to_lambda_tildes, lambda_tildes_to_lambdas
from jimgw.prior import Powerlaw, Uniform, Composite
from jimgw.waveform import RippleTaylorF2
from astropy.time import Time
from jimgw.detector import H1, L1, V1

from injection_recovery import HYPERPARAMETERS, PRIOR, NAMING, generate_params_dict, compute_snr
import corner
from scipy.stats import percentileofscore
# Force CPU for the postprocessing scipt
jax.config.update("jax_platform_name", "cpu")
print(jax.devices())
from tqdm import tqdm

### Script constants
use_chi_eff = False # whether to use chi_eff instead of chi_1, chi_2
convert_lambdas = False # whether to convert lambda1, lambda2 to lambda_tilde, delta_lambda_tilde
ignore_tc = False # whether to ignore t_c in the plots
reweigh_distance = False # whether to reweigh samples based on the distance
outdir = "/home/thibeau.wouters/public_html/jim_injections/tests_TaylorF2_31_01_2024/"
postprocessing_dir = "./postprocessing/"

### Hyperparameters
params = {
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
plt.rcParams.update(params)

corner_label_fs = 24
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=corner_label_fs),
                        title_kwargs=dict(fontsize=corner_label_fs), 
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

# Remove t_c from labels if ignored
if ignore_tc:
    labels_tidal_deltalambda.remove(r'$t_c$')
    labels_tidal_lambda12.remove(r'$t_c$') 
    labels_tidal_deltalambda_chi_eff.remove(r'$t_c$')
    labels_tidal_lambda12_chi_eff.remove(r'$t_c$')
    
# Choose one as basic version
labels_tidal = labels_tidal_lambda12 

#################
### Debugging ###
#################

def get_injected_snr(print_sorted: bool = True, arrow_idx_list: list = []):
    """
    Small function to get the injected network SNR for each injection, and print it to the screen.
    
    Args:
        arrow_idx_list (list): A list of indices of injections that get marked with an arrow in the print (e.g. for debug).
    """
    
    result = []
    
    # Iterate over the subdirectories
    
    for subdir in os.listdir(outdir):
        subdir_path = os.path.join(outdir, subdir)
        if os.path.isdir(subdir_path):
            # Load snr.csv
            snr_path = os.path.join(subdir_path, "snr.csv")
            df = pd.read_csv(snr_path)
            # Get injected network SNR
            snr_values = df["snr"].values
            network_snr = snr_values[-1]
            result.append(network_snr)
            
    # Show the results
    if not print_sorted:
        for i, snr in enumerate(result):
            print(f"{i}: {snr}")
    
    # Also print sorted results if wanted
    else:
        # Now sort the result list and print it again
        result = np.array(result)
        sort_idx = np.argsort(result)
        for i in sort_idx:
            if i in arrow_idx_list:
                my_string = " <---"
            else:
                my_string = ""
            snr = result[i]
            print(f"{i}: {snr} {my_string}")
        
    
###############################
### Post-injection analysis ###
###############################

def compute_number_correct(true: np.array, low: np.array, high: np.array) -> np.array:
    """
    TODO: This is not used anywhere, so remove it?
    Compute which of the recovered params are within a given credible interval of the true value.

    Args:
        true (np.array): True parameter values
        low (np.array): Lower bound of credible interval computed from injection recovery
        high (np.array): Upper bound of credible interval computed from injection recovery

    Returns:
        np.array: Array of booleans indicating whether that parameter was inside the credible interval or not.
    """
    
    mask = np.logical_and(low <= true, true <= high)
    correct_list = np.where(mask, True, False)
    return correct_list

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
                                       return_first: bool = False,
                                       two_sided: bool = False) -> tuple[np.array, float]:
    """
    Finds the true parameter set from a list of true parameter sets, and also computes its credible level.

    Args:
        true_params_list (np.array): List of true parameters and copies for sky location.

    Returns:
        tuple[np.array, float]: The select true parameter set, and its credible level.
    """
    
    if return_first:
        # Ignore the sky location mirrors, just take the first one
        true_params = true_params_list[0]
        true_params_list = [true_params]
    
    for i, true_params in enumerate(true_params_list):
        params_credible_level_list = []
        
        ### OLD code
        # boolean_values = np.array(chains < true_params)
        # credible_level = np.mean(boolean_values, axis = 0)
        
        for j, param in enumerate(true_params):
            q = percentileofscore(chains[:, j], param)
            q /= 100.0
            if two_sided:
                credible_level = 1 - 2 * min(q, 1-q)
            else:
                credible_level = q
                
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
    

def get_credible_levels_injections(reweigh_distance: bool = False,
                                   return_first: bool = False,
                                   snr_cutoff: float = 0,
                                   plot_snr_vs_credible: bool = True,
                                   show_q_sorted: bool = True,
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
    snr_values_list = []
    
    subdir_list = []
    
    counter = 0
    
    for subdir in os.listdir(outdir):
        subdir_path = os.path.join(outdir, subdir)
        
        if os.path.isdir(subdir_path):
            # Load the snr.csv and get network SNR:
            snr_path = os.path.join(subdir_path, "snr.csv")
            df = pd.read_csv(snr_path)
            # Get injected network SNR
            snr_values = df["snr"].values
            network_snr = snr_values[-1]
            
            if network_snr < snr_cutoff:
                print(f"INFO: Skipping injection {subdir} with SNR < cutoff: {network_snr}")
                continue
            
            # Note: we skip the iteration if the config json file or production samples files does not exist (e.g. if the injection run failed)
            json_path = os.path.join(subdir_path, "config.json")
            chains_filename = f"{subdir_path}/results_production.npz"
            if not os.path.isfile(json_path) or not os.path.isfile(chains_filename):
                continue
            
            snr_values_list.append(network_snr)
            subdir_list.append(subdir)
            
            # Load config, and get the injected parameters
            with open(json_path, "r") as f:
                config = json.load(f)
                
            # TODO make this less cumbersome
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
            
            # # Get the sky mirrored values as well, NOTE this outputs an array of 2/3 arrays!
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
    
    if plot_snr_vs_credible:
        snr_values = np.array(snr_values_list) # get chirp mass
        cred = credible_level_list[:, 6]
        # Plot the snr values vs credible levels
        plt.figure(figsize=(12, 9))
        plt.plot(snr_values, cred, "o", color="blue")
        plt.xlabel("SNR")
        plt.ylabel("Credible level")
        plt.savefig("./postprocessing/snr_vs_q.png", bbox_inches="tight")
        plt.close()
        
    if show_q_sorted:
        # We sort here based on the credible level of d_L
        cred = credible_level_list[:, 6]
        cred = np.array(cred)
        sort_idx = np.argsort(cred)#[::-1]
        for i in sort_idx:
            cred_val = cred[i]
            subdir = subdir_list[i]
            print(f"{subdir}: {cred_val}")
    
    return credible_level_list


def make_pp_plot(credible_level_list: np.array, 
                 percentile_list: list = [0.68, 0.95, 0.995], 
                 nb_bins: int = 100) -> None:
    """
    Creates a pp plot from the credible levels.

    Args:
        credible_level_list (np.array): List of credible levels obtained from injections.
        percentile (float/list, optional): Percentile used for upper and lower quantile. Defaults to 0.05.
        nb_bins (int, optional): Number of bins in the histogram. Defaults to 100.
    """
    
    # Group the plotting hyperparameters here: 
    bbox_to_anchor = (1.1, 1.0)
    legend_fontsize = 20
    handlelength = 1
    linewidth = 2
    min_alpha = 0.05
    max_alpha = 0.15
    shadow_color = "blue"
    n = np.shape(credible_level_list)[1]
    color_list = cm.rainbow(np.linspace(0, 1, n))
    
    # First, get uniform distribution cumulative histogram:
    nb_injections, n_dim = np.shape(credible_level_list)
    print("nb_injections, n_dim")
    print(nb_injections, n_dim)
    N = 10_000
    uniform_histogram = make_uniform_cumulative_histogram((N, nb_injections), nb_bins=nb_bins)
    
    # Check if given percentiles is float or list
    if isinstance(percentile_list, float):
        percentile_list = [percentile_list]
    # Convert the percentages
    percentile_list = [1 - p for p in percentile_list]
    # Get list of alpha values:
    alpha_list = np.linspace(min_alpha, max_alpha, len(percentile_list))
    alpha_list = alpha_list[::-1]
        
    plt.figure(figsize=(12, 9))    
    # Plot the shadow bands
    for percentile, alpha in zip(percentile_list, alpha_list):
        
        upper_quantile_array = []
        lower_quantile_array = []
        for i in range(nb_bins):
            upper_quantile_array.append(np.quantile(uniform_histogram[:, i], (1-percentile/2)))
            lower_quantile_array.append(np.quantile(uniform_histogram[:, i], (percentile/2)))
        
        bins = np.linspace(0, 1, nb_bins + 1)
        bins = (bins[1:]+bins[:-1])/2
        plt.fill_between(bins, lower_quantile_array, upper_quantile_array, color = shadow_color, alpha = alpha)
    
    # Compute the x data for the plot
    x = np.append(0, bins)
    # Will save the computed pvalues here
    pvalues = []
    print("Creating pp-plot, getting p values . . .")
    for i, label in enumerate(labels_tidal):
        # Compute the p-value
        p = kstest(credible_level_list[:nb_injections, i], cdf = uniform(0,1).cdf).pvalue
        # Compute the y data for the plot
        y = np.append(0, make_cumulative_histogram(credible_level_list[:nb_injections, i]))
        col = color_list[i]
        plt.plot(x, y, c=col, label = f"{label} ($p = {p:.2f}$) ", linewidth = linewidth)
        pvalues.append(p)
    plt.legend(bbox_to_anchor = bbox_to_anchor, fontsize = legend_fontsize, handlelength=handlelength)
    plt.xlabel(r'confidence level')
    plt.ylabel(r'fraction of samples with confidence level $\leq x$')
    print("Creating pp-plot, getting p values . . . DONE")

    print("pvalues")
    print(pvalues)
    ptotal = kstest(pvalues, cdf=uniform(0,1).cdf).pvalue
    string_total = f"Total p-value: {ptotal:.2f}"
    print(string_total)
    
    # TODO debug this
    string_total = "" 
    
    print("Saving pp-plot")
    plt.grid(False) # disable grid
    plt.title(string_total)
    save_name = outdir + "pp_plot"
    for ext in [".png", ".pdf"]:
        full_savename = save_name + ext
        plt.savefig(full_savename, bbox_inches="tight")
        print(f"Saved pp-plot to: {full_savename}")
        
def make_snr_histogram(bins=20) -> None:
    """
    Creates a histogram of SNR values of injections
    """
    
    # Go over all the subdirectories and get the SNR values
    snr_values = []
    filenames = []
    for subdir in os.listdir(outdir):
        subdir_path = os.path.join(outdir, subdir)
        if os.path.isdir(subdir_path):
            # Load snr.csv
            snr_path = os.path.join(subdir_path, "snr.csv")
            filenames.append(snr_path)
            df = pd.read_csv(snr_path)
            # Get injected network SNR
            snr_val = df["snr"].values[-1]
            snr_values.append(snr_val)
            
    snr_values = np.array(snr_values)
    
    # Make histogram and plot
    plt.figure(figsize=(12, 9))
    plt.hist(snr_values, bins=bins, histtype="step", color="blue", linewidth=2, density=True)
    save_path = outdir + "snr_histogram.png"
    print(f"Saving snr histogram to: {save_path}")
    plt.xlabel("SNR")
    plt.ylabel("Density")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    # Sort the injections based on SNR, from highest to lowest
    sort_idx = np.argsort(snr_values)[::-1]
    print("SNR values sorted:")
    sorted_snr_values = snr_values[sort_idx]
    print(sorted_snr_values)
    
    # also print the indices
    print("Indices sorted:")
    filenames = np.array(filenames)
    sorted_filenames = filenames[sort_idx]
    print(sorted_filenames)
    
    # iterate over the sorted filenames, load its config, and print the q param
    q_list = []
    for filename in sorted_filenames:
        filename = filename.replace("snr.csv", "config.json")
        with open(filename, "r") as f:
            config = json.load(f)
        q_list.append(config["q"])
        
    # Plot the q values vs SNR values
    plt.plot(sorted_snr_values, q_list, "o", color="blue")
    plt.xlabel("SNR")
    plt.ylabel("q")
    plt.savefig("./postprocessing/snr_vs_q.png", bbox_inches="tight")
    plt.close()
        
    
def make_dL_histogram(bins=20) -> None:
    """
    Creates a histogram of dL values of injections
    """
    
    # Go over all the subdirectories and get the SNR values
    dL_values = []
    for subdir in os.listdir(outdir):
        subdir_path = os.path.join(outdir, subdir)
        if os.path.isdir(subdir_path):
            # Load config.json
            config = json.load(open(subdir_path + "/config.json", "r"))
            dL = config["d_L"]
            dL_values.append(dL)
            
    dL_values = np.array(dL_values)
    
    # Generate a histogram of the dL values
    dL_prior = Powerlaw(30.0, 300.0, alpha = 2.0, naming=["d_L"])
    N = 10_000
    # Generate the dL sample from a Powerlaw prior instead of a uniform prior
    seed = np.random.randint(low=0, high=10000)
    key, subkey = jax.random.split(jax.random.PRNGKey(seed + 42))
    samples = dL_prior.sample(subkey, N)["d_L"]
    samples = np.array(samples)
    print("samples")
    print(samples)
    
    print(len(dL_values))
    
    # Make histograms
    counts_samples, bins = np.histogram(samples, bins = bins, density=True)
    counts_injections, _ = np.histogram(dL_values, bins = bins, density=True)
    
    # Plot histograms
    plt.figure(figsize=(12, 9))
    plt.stairs(counts_samples, bins, linewidth=2, label="Powerlaw")
    plt.stairs(counts_injections, bins, linewidth=2, label="Injection samples")
    save_path = outdir + "dL_distribution.png"
    print(f"Saving dL histogram to: {save_path}")
    plt.xlabel(r"$d_L$")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
        
# def count_successful_injections(outdir: str) -> int:
#     """
#     Count the number of successful injections in a given directory.
#     If a subdirectory has saved a results_production.npz file, we consider it a successful injection.

#     Args:
#         outdir (str): The outdir to which all injection results were saved.

#     Returns:
#         int: Number of injection dirs for which we have saved a results_production.npz file.
#     """
    
#     counter = 0
#     for subdir in os.listdir(outdir):
#         subdir_path = os.path.join(outdir, subdir)
#         if os.path.isdir(subdir_path):
#             chains_filename = f"{subdir_path}/results_production.npz"
#             if os.path.isfile(chains_filename):
#                 counter += 1
                
#     return counter
            
            
################
### Plotting ###
################

def compute_chi_eff(mc, q, chi1, chi2):
    eta = q/(1+q)**2
    m1, m2 = Mc_eta_to_ms(jnp.array([mc, eta]))
    params = jnp.array([m1, m2, chi1, chi2])
    chi_eff = get_chi_eff(params)
    
    return chi_eff

def plot_chains_and_true_value(N_config: int = 0, 
                               is_tidal: bool = True,
                               ignore_tc: bool = False,
                               check_mirror_locations: bool = True,
                               reweigh_distance: bool = False):
    """
    Creates a plot showing the posterior samples and the true, i.e. injected, values on top of it.
    
    The plot will be saved to the same directory as where the samples are loaded.

    Args:
        N_config (int/list, optional): Number of the injection. Defaults to 0.
        is_tidal (bool, optional): Whether this is for tidal parameters. Defaults to True.
        ignore_tc (bool, optional): Whether t_c should be ignored in the plot or not. Defaults to False.
    """
    
    # If single integer is given, convert to list
    if isinstance(N_config, int):
        N_config = [N_config]
        
    # If N_config is None, then create list of all injection directories, but check if they have production samples saved
    if N_config is None or N_config == []:
        N_config = []
        for subdir in os.listdir(outdir):
            subdir_path = os.path.join(outdir, subdir)
            if os.path.isdir(subdir_path):
                chains_filename = f"{subdir_path}/results_production.npz"
                if os.path.isfile(chains_filename):
                    # Chains exist, so extract the config number and save it
                    match = re.search(r'injection_(\d+)', subdir_path)
                    if match:
                        number = int(match.group(1))
                        N_config.append(number)
                    else:
                        print("WARNING -- No number found in the expression.")
                    
        
    # Sort N_config
    N_config = sorted(N_config)
    print("Running postprocessing for the following injections:")
    print(N_config)
        
    for N in N_config:
        print(f"--- Creating plot for injection {N} . . .")
        # Get the specific injection directory
        this_outdir = outdir + f"injection_{N}/"
        corner_kwargs = default_corner_kwargs

        print(f"Reading data from {this_outdir}")
        # These indices are in case t_c is deleted
        iota_index = 8 
        delta_index = 11
        if ignore_tc:
            idx_list = [0,1,2,3,4,5,6,8,9,10,11,12]
        else:
            idx_list = [0,1,2,3,4,5,6,7,8,9,10,11,12]
            # add one to the indices if t_c is there
            iota_index += 1
            delta_index += 1

        print(f"Creating plots with use_chi_eff={use_chi_eff}")

        filename = f"{this_outdir}results_production.npz"

        print(f"Loading production samples... ")
        data = np.load(filename)
        if ignore_tc:
            chains = data['chains'][:,:,idx_list].reshape(-1,12)
        else:
            chains = data['chains'].reshape(-1,13)
         
        chains[:, iota_index] = np.arccos(chains[:, iota_index])
        chains[:, delta_index] = np.arcsin(chains[:, delta_index])
            
        chains = np.asarray(chains)
        print("Loading chains complete")
        
        ### Load the injection params
        print("Loading injection params")
        config_file = this_outdir + 'config.json'
        # Load the config file
        with open(config_file, 'r') as f:
            config = json.load(f)
        # Load the injection params
        # Remove t_c from naming
        naming = copy.deepcopy(NAMING)
        if ignore_tc:
            naming.remove('t_c')
            
        # Get the index of d_L
        d_L_index = naming.index("d_L")
        if reweigh_distance:
            # Reweigh the distance
            print("INFO: Reweighing distance. . .")
            d_values = chains[:, d_L_index]
            weights = d_values**2
        else:
            weights = None
            
        true_params = np.array([config[param_name] for param_name in naming])
        if check_mirror_locations:
            old_true_params = copy.deepcopy(true_params)
            print("True params from the config:")
            print(old_true_params)
            print("INFO: Will check the mirror locations as well to find true parameters. . .")
            # Find true params, based on mirror location
            mirrored_values = get_mirror_location(true_params)
            mirrored_values = np.vstack(mirrored_values)
            all_true_params = np.vstack((true_params, mirrored_values))
            true_params, _ = get_true_params_and_credible_level(chains, all_true_params)
            # Check if true params is same as old params or not
            if np.allclose(true_params, old_true_params):
                print("True params are the same as before!")
            else:
                print("True params are different from before!")
            print("True params after getting sky location:")
            print(true_params)
        print("Loading injection params complete")

        ### Plotting    
        name = this_outdir + f"injection_results.png" 
        print(f"Saving plot of chains to {name}")
        
        if convert_lambdas:
            lambda1, lambda2 = chains[:,4], chains[:,5]
            mc, q = chains[:,0], chains[:,1]
            eta = q/(1+q)**2
            m1, m2 = Mc_eta_to_ms(jnp.array([mc, eta]))
            lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(jnp.array([lambda1, lambda2, m1, m2]))
            chains[:,4] = lambda_tilde
            chains[:,5] = delta_lambda_tilde
        
        if use_chi_eff:
                
            # For the samples:
            mc, q, chi1, chi2 = chains[:,0], chains[:,1], chains[:,2], chains[:,3]
            chi_eff = compute_chi_eff(mc, q, chi1, chi2)
            chains = np.delete(chains, [2,3], axis=1)
            chains = np.insert(chains, 2, chi_eff, axis=1)
            
            # For the true values
            mc, q, chi1, chi2 = true_params[0], true_params[1], true_params[2], true_params[3]
            chi_eff = compute_chi_eff(mc, q, chi1, chi2)
            true_params = np.delete(true_params, [2,3])
            true_params = np.insert(true_params, 2, chi_eff)
            
        # Define the labels
        if not is_tidal:
            print("Not tidal -- labels not defined for this?")
            labels = None
        else:
            if not convert_lambdas:
                labels = labels_tidal_lambda12_chi_eff if use_chi_eff else labels_tidal_lambda12
            else:
                labels = labels_tidal_deltalambda_chi_eff if use_chi_eff else labels_tidal_deltalambda
        
        # Finally, make the plot:
        true_params = list(true_params) 
        fig = corner.corner(chains, labels = labels, weights=weights, truths=true_params, truth_color="red", hist_kwargs={'density': True}, **corner_kwargs)
        fig.savefig(name, bbox_inches='tight')  
        print("Done")
    
### 
### OTHER 
###

def generate_gw_parameters(N: int = 1000, snr_threshold=12) -> np.array:

    # End result is going to be saved here
    my_dict = {}
    
    #M Main computation
    naming = list(PRIOR.keys())
    prior_ranges = jnp.array([PRIOR[name] for name in naming])
    prior_low, prior_high = prior_ranges[:, 0], prior_ranges[:, 1]
    bounds = []
    for key, value in PRIOR.items():
        bounds.append(value)

    bounds = np.asarray(bounds)
    xmin = bounds[:, 0]
    xmax = bounds[:, 1]
    
    for name in naming:
        my_dict[name] = []
    
    for i in tqdm(range(N)):
        network_snr = 0
        
        while network_snr < snr_threshold:
            config = generate_params_dict(prior_low, prior_high, naming)
            # Check the SNR
                    ### Data definitions

            gps = 1187008882.43
            trigger_time = gps
            gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad
            fmin = 20
            f_ref = fmin 
            fmax = 2048
            f_sampling = 2 * fmax
            T = 256
            duration = T
            post_trigger_duration = 2

            ### Define priors
            
            # Internal parameters
            Mc_prior = Uniform(xmin[0], xmax[0], naming=["M_c"])
            q_prior = Uniform(
                xmin[1], 
                xmax[1],
                naming=["q"],
                transforms={"q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2)},
            )
            s1z_prior                = Uniform(xmin[2], xmax[2], naming=["s1_z"])
            s2z_prior                = Uniform(xmin[3], xmax[3], naming=["s2_z"])
            
            first_lambda_prior  = Uniform(xmin[4], xmax[4], naming=["lambda_1"])
            second_lambda_prior = Uniform(xmin[5], xmax[5], naming=["lambda_2"])

            # External parameters
            # dL_prior       = Powerlaw(xmin[6], xmax[6], 2.0, naming=["d_L"])
            dL_prior       = Uniform(xmin[6], xmax[6], naming=["d_L"])
            t_c_prior      = Uniform(xmin[7], xmax[7], naming=["t_c"])

            # These priors below are always the same, no xmin and xmax needed
            phase_c_prior  = Uniform(0.0, 2 * jnp.pi, naming=["phase_c"])
            cos_iota_prior = Uniform(
                -1.0,
                1.0,
                naming=["cos_iota"],
                transforms={
                    "cos_iota": (
                        "iota",
                        lambda params: jnp.arccos(
                            jnp.arcsin(jnp.sin(params["cos_iota"] / 2 * jnp.pi)) * 2 / jnp.pi
                        ),
                    )
                },
            )
            psi_prior     = Uniform(0.0, jnp.pi, naming=["psi"])
            ra_prior      = Uniform(0.0, 2 * jnp.pi, naming=["ra"])
            sin_dec_prior = Uniform(
                -1.0,
                1.0,
                naming=["sin_dec"],
                transforms={
                    "sin_dec": (
                        "dec",
                        lambda params: jnp.arcsin(
                            jnp.arcsin(jnp.sin(params["sin_dec"] / 2 * jnp.pi)) * 2 / jnp.pi
                        ),
                    )
                },
            )

            ### Compose the prior
            prior_list = [
                    Mc_prior,
                    q_prior,
                    s1z_prior,
                    s2z_prior,
                    first_lambda_prior,
                    second_lambda_prior,
                    dL_prior,
                    t_c_prior,
                    phase_c_prior,
                    cos_iota_prior,
                    psi_prior,
                    ra_prior,
                    sin_dec_prior,
            ]
            prior = Composite(prior_list)
            bounds = jnp.array([[p.xmin, p.xmax] for p in prior.priors]).T
            naming = NAMING
            
                ### Get the injected parameters, but apply the transforms first
            true_params_values = jnp.array([config[name] for name in naming])

            true_params = dict(zip(naming, true_params_values))

            # Apply prior transforms to this list:
            true_params = prior.transform(true_params)

            # Convert values from single float arrays to just float
            true_params = {key: value.item() for key, value in true_params.items()}
            T = 256
            duration = T
            epoch = duration - post_trigger_duration
            freqs = jnp.linspace(fmin, fmax, duration * f_sampling)


            ### Getting ifos and overwriting with above data

            detector_param = {"ra": true_params["ra"], 
                            "dec": true_params["dec"], 
                            "gmst": gmst, 
                            "psi": true_params["psi"], 
                            "epoch": epoch, 
                            "t_c": true_params["t_c"]}

            ### Inject signal, fetch PSD and overwrite

            waveform = RippleTaylorF2(f_ref=f_ref)
            h_sky = waveform(freqs, true_params)
            
            seed = np.random.randint(low=0, high=10000)
            key, subkey = jax.random.split(jax.random.PRNGKey(seed + 1234))
            H1.inject_signal(subkey, freqs, h_sky, detector_param, psd_file = "psd.txt", no_noise=False)
            key, subkey = jax.random.split(key)
            L1.inject_signal(subkey, freqs, h_sky, detector_param, psd_file = "psd.txt", no_noise=False)
            key, subkey = jax.random.split(key)
            V1.inject_signal(subkey, freqs, h_sky, detector_param, psd_file = "psd_virgo.txt", no_noise=False)
            
            h1_snr = compute_snr(H1, h_sky, detector_param)
            l1_snr = compute_snr(L1, h_sky, detector_param)
            v1_snr = compute_snr(V1, h_sky, detector_param)
            
            network_snr = np.sqrt(h1_snr**2 + l1_snr**2 + v1_snr**2)
            
            # print("Network SNR:", network_snr)
        
        # If we get out of the main loop, we have network SNR above the threshold, so add to my_dict
        for name in naming:
            my_dict[name].append(config[name])
            
    return my_dict

def check_credible_level_list(credible_level_list):
    
    naming = list(PRIOR.keys())
    
    print("np.shape(credible_level_list)")
    print(np.shape(credible_level_list))
    
    for i, name in enumerate(naming):
        print(name)
        samples = credible_level_list[:, i]
        hist, bin_edges = np.histogram(samples, bins=20, density=True)
        plt.stairs(hist, bin_edges, linewidth=2)
        plt.title(name)
        plt.savefig(f"./postprocessing/{name}_credible_levels.png")
        plt.close()
        
        # Get a cumsum of the samples's density
        cumsum = np.cumsum(hist) / np.sum(hist)
        plt.plot(bin_edges[1:], cumsum)
        plt.plot([0,1], [0,1], color="black")
        plt.title(name)
        plt.savefig(f"./postprocessing/{name}_credible_levels_cumsum.png")
        plt.close()


def make_tc_posterior_histogram(nb_bins = 20, alpha = 0.1):
    
    # Iterate over all subdirectories in OUTDIR, and load the results_production.npz file if there
    print("Reading data ... ")
    tc_values = []
    for subdir in os.listdir(outdir):
        subdir_path = os.path.join(outdir, subdir)
        if os.path.isdir(subdir_path):
            chains_filename = f"{subdir_path}/results_production.npz"
            if os.path.isfile(chains_filename):
                # Load the chains
                data = np.load(chains_filename)
                chains = data['chains'].reshape(-1,13)
                # Get the tc values
                tc_values.append(chains[:, 7])    
                
    tc_values = np.array(tc_values)
    print("Reading data ... DONE")
    
    # Plot all these histograms on top of each other
    plt.figure(figsize=(12, 9))
    for tc in tc_values:
        hist, edges = np.histogram(tc, bins=nb_bins, density=True)
        plt.stairs(hist, edges, linewidth=2, alpha=alpha, color = "blue")
    plt.xlabel("t_c")
    plt.ylabel("Density")
    plt.axvline(x=0, color="black", linestyle="--")
    plt.savefig("./postprocessing/tc_posterior_histogram.png", bbox_inches="tight")
    plt.close()
        
        
def plot_accs(which = "global_accs", alpha = 0.2, single_number = True):
    """
    GO over all the injections and plot all the local or global accs on top of each other, or make histogram of final mean value

    Args:
        which (str, optional): Plot global or local accs.. Defaults to "global_accs".
        alpha (float, optional): Alpha for plot in case we plot curves. Defaults to 0.2.
        single_number (bool, optional): Plot curves or summarize run by mean of accs. Defaults to True.
    """
    
    accs = []
    print("Reading data ... ")
    for subdir in os.listdir(outdir):
        subdir_path = os.path.join(outdir, subdir)
        if os.path.isdir(subdir_path):
            # Check if the results_production.npz file exists otherwise continue
            if not os.path.isfile(f"{subdir_path}/results_production.npz"):
                continue
            chains_filename = f"{subdir_path}/results_production.npz"
            if os.path.isfile(chains_filename):
                # Load the chains
                data = np.load(chains_filename)
                this_accs = data[which]
                this_accs = np.mean(this_accs, axis = 0)
                
                if single_number:
                    this_accs = np.mean(this_accs)
                
                accs.append(this_accs)
            
    print("Reading data ... DONE")    
    accs = np.array(accs)
    
    # Plot all these histograms on top of each other
    plt.figure(figsize=(12, 9))
    if single_number:
        # Plot the histogram
        plt.hist(accs, bins=40, density=True, linewidth=2, color = "blue", histtype="step")
        plt.xlabel(which)
        plt.ylabel("Density")
    else:
        # plot the individual curves
        for acc in accs:
            plt.plot(acc, color="blue", alpha=alpha)
        plt.xlabel(which)
        plt.ylabel("Iteration")
    plt.savefig(f"./postprocessing/{which}.png", bbox_inches="tight")
    plt.close()
    
def plot_gammas():
    return 
    
#############
### Main ####
#############
        
if __name__ == "__main__":
    
    ### Choose what to do here
    do_postprocessing_plots = True
    do_pp_plot = True
    do_snr_hist = False
    do_dL_hist = False
    do_tc_hist = False
    check_gw_params_distributions = False
    plot_gw_params_distribution = False
    do_global_accs = False
    do_check_gammas = False
    
    if do_postprocessing_plots:
        plot_chains_and_true_value(N_config = None, 
                                   check_mirror_locations = False,
                                   reweigh_distance = False)
    
    if do_pp_plot:
        print("Getting credible_level_list . . .")
        credible_level_list = get_credible_levels_injections(reweigh_distance = reweigh_distance, 
                                                             return_first = True, 
                                                             snr_cutoff = 0,
                                                             plot_snr_vs_credible=True,
                                                             show_q_sorted=True,
                                                             max_number_injections=-1)
        print("Getting credible_level_list . . . DONE")
        
        print("Checking credible level list")
        check_credible_level_list(credible_level_list)
        
        print(np.shape(credible_level_list))
        
        make_pp_plot(credible_level_list)
        
    if do_snr_hist:
        make_snr_histogram()
        
    if do_dL_hist:
        make_dL_histogram()
    
    if do_tc_hist:
        make_tc_posterior_histogram()
        
    if do_global_accs:
        plot_accs("global_accs", single_number=True)
        plot_accs("local_accs", single_number=True)
        
    if do_check_gammas:
        print("NOTE: still have to implement checking the adaptive gammas. . .")
        plot_gammas()
        
    if check_gw_params_distributions:
        result = generate_gw_parameters(10_000)
        # Save this dictionary:
        save_path = "./postprocessing/injection_set.json" 
        with open(save_path, "w") as f:
            json.dump(result, f)
        print(f"Saved injection set to {save_path}")
        
    if plot_gw_params_distribution:
        # Load the injection parameters
        save_path = "./postprocessing/injection_set.json" 
        with open(save_path, "r") as f:
            result = json.load(f)
        
        # Plot the distributions
        dL = np.array(result["d_L"])
        
        # Make histogram
        for key, value in result.items():
            value = np.array(value)
            plt.figure(figsize=(12, 9))
            plt.hist(value, bins=50, histtype="step", color="blue", linewidth=2, density=True)
            save_path = f"./postprocessing/{key}_histogram.png"
            print(f"Saving {key} histogram to: {save_path}")
            plt.xlabel(key)
            plt.ylabel("Density")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            
        ### Also compare dL to the prior
        
        # dL_prior = Powerlaw(30.0, 300.0, alpha = 2.0, naming=["d_L"])
        dL_prior = Uniform(30.0, 300.0, naming=["d_L"])
        N = 10_000
        # Generate the dL sample from a Powerlaw prior instead of a uniform prior
        seed = np.random.randint(low=0, high=10000)
        key, subkey = jax.random.split(jax.random.PRNGKey(seed + 42))
        samples = dL_prior.sample(subkey, N)["d_L"]
        samples = np.array(samples)
        
        injection_samples = result["d_L"]
        injection_samples = np.array(injection_samples)
        
        counts_samples, bins = np.histogram(samples, bins = 50, density=True)
        counts_injections, _ = np.histogram(injection_samples, bins = bins, density=True)
        
        plt.figure(figsize=(12, 9))
        plt.stairs(counts_samples, bins, linewidth=2, label="Powerlaw")
        plt.stairs(counts_injections, bins, linewidth=2, label="Injection samples")
        save_path = f"./postprocessing/compare_dL_distributions.png"
        print(f"Saving {key} histogram to: {save_path}")
        plt.xlabel("d_L")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        
    
    print("Done.")