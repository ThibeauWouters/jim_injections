import os
os.environ["CUDA_VISIBILE_DEVICES"] = "-1"
import psutil
p = psutil.Process()
p.cpu_affinity([0])
import numpy as np
# import arviz 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.stats import kstest, uniform, percentileofscore

from ripple import get_chi_eff, Mc_eta_to_ms, lambdas_to_lambda_tildes, lambda_tildes_to_lambdas
from jimgw.prior import Uniform, Composite
from jimgw.single_event.waveform import RippleTaylorF2
from jimgw.single_event.detector import H1, L1, V1
from astropy.time import Time

import utils
import importlib
importlib.reload(utils)
plt.rcParams.update(utils.matplotlib_params)

### Script constants
reweigh_distance = False # whether to reweigh samples based on the distance
outdir = "../tidal/outdir_TaylorF2_part3/"
postprocessing_dir = "./pp_TaylorF2/"
    
###############################
### Post-injection analysis ###
###############################


def make_pp_plot(credible_level_list: np.array, 
                 which_percentile_calculation,
                 percentile_list: list = [0.68, 0.95, 0.995], 
                 nb_bins: int = 100,
                 params_idx: list = None,
                 reweigh_distance: bool = False,
                 ) -> None:
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
    shadow_color = "gray"
    n = np.shape(credible_level_list)[1]
    color_list = cm.rainbow(np.linspace(0, 1, n))
    
    # First, get uniform distribution cumulative histogram:
    nb_injections, n_dim = np.shape(credible_level_list)
    print("nb_injections, n_dim")
    print(nb_injections, n_dim)
    N = 10_000
    uniform_histogram = utils.make_uniform_cumulative_histogram((N, nb_injections), nb_bins=nb_bins)
    
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
    if params_idx is None:
        params_idx = range(n_dim)
        
    for i in params_idx: 
        label = utils.labels_tidal[i]
        # Compute the p-value
        p = kstest(credible_level_list[:nb_injections, i], cdf = uniform(0,1).cdf).pvalue
        # Compute the y data for the plot
        y = np.append(0, utils.make_cumulative_histogram(credible_level_list[:nb_injections, i]))
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
    string_total = f"N = {len(credible_level_list)}, Total p-value: {ptotal:.2f}"
    print(string_total)
    
    # TODO add p value or not?
    title_string = f"N = {len(credible_level_list)}"
    print(title_string)
    
    print("Saving pp-plot")
    plt.grid(False) # disable grid
    plt.title(title_string)
    save_name = outdir + "pp_plot_" + which_percentile_calculation
    if reweigh_distance:
        save_name += "_reweighed"
        
    for ext in [".png", ".pdf"]:
        full_savename = save_name + ext
        print(f"Saving pp-plot to: {full_savename}")
        plt.savefig(full_savename, bbox_inches="tight")
        
#############
### Main ####
#############
        
if __name__ == "__main__":
    
    which_percentile_calculation = "two_sided" # which function to use to compute the percentiles
    reweigh_distance = False # whether to reweigh samples based on the distance, due to the SNR cutoff used
    return_first = True # whether to just use injected params or to also look at sky mirrored locations
    convert_cos_sin = True
    thinning_factor = 100
    
    print(f"script hyperparams: \nreweigh_distance = {reweigh_distance}, \nreturn_first = {return_first}, \nconvert_cos_sin = {convert_cos_sin}")
    
    kde_file = "../postprocessing/kde_dL.npz"
    data = np.load(kde_file)
    kde_x = data["x"]
    kde_y = data["y"]

    if reweigh_distance:
        weight_fn = lambda x: np.interp(x, kde_x, kde_y)
    else:
        weight_fn = lambda x: x
    
    print(f"which_percentile_calculation is set to {which_percentile_calculation}")
    credible_levels, subdirs = utils.get_credible_levels_injections(outdir, 
                                                                    return_first=return_first,
                                                                    weight_fn=weight_fn,
                                                                    thinning_factor=thinning_factor,
                                                                    which_percentile_calculation=which_percentile_calculation,
                                                                    save=False,
                                                                    convert_cos_sin=convert_cos_sin)
    make_pp_plot(credible_levels, 
                 which_percentile_calculation=which_percentile_calculation, 
                 reweigh_distance=reweigh_distance)
    
    # Sanity checking:
    mc_credible_levels = credible_levels[:, 0]
    npositive = np.sum(np.where(mc_credible_levels > 0.5, 1, 0))
    print(f"{npositive} positive credible levels out of {len(mc_credible_levels)} samples.")