import os
import matplotlib.pyplot as plt
import json
import numpy as np
import jax.numpy as jnp
from jimgw.single_event.detector import Detector
from jimgw.single_event.likelihood import SingleEventLiklihood, HeterodynedTransientLikelihoodFD
import corner
import jax
from jaxtyping import Array, Float

from injection_recovery import NAMING

NAMING = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']

default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False,
                        truth_color="red")

matplotlib_params = {"axes.grid": True,
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

plt.rcParams.update(matplotlib_params)


############################################
### Injection recovery utility functions ###
############################################

def compute_snr(detector: Detector, h_sky: dict, detector_params: dict):
    """Compute the SNR of an event for a single detector, given the waveform generated in the sky.

    Args:
        detector (Detector): Detector object from jim.
        h_sky (dict): Dict of jax numpy array containing the waveform strain as a function of frequency in the sky frame
        detector_params (dict): Dictionary containing parameters of the event relevant for the detector.
    """
    frequencies = detector.frequencies
    df = frequencies[1] - frequencies[0]
    align_time = jnp.exp(
        -1j * 2 * jnp.pi * frequencies * (detector_params["epoch"] + detector_params["t_c"])
    )
    
    waveform_dec = (
                detector.fd_response(detector.frequencies, h_sky, detector_params) * align_time
            )
    
    snr = 4 * jnp.sum(jnp.conj(waveform_dec) * waveform_dec / detector.psd * df).real
    snr = jnp.sqrt(snr)
    return snr

def generate_params_dict(prior_low: jnp.array, prior_high: jnp.array, params_names: dict) -> dict:
    """
    Generate a dictionary of parameters from the prior range.

    Args:
        prior_low (jnp.array): Lower bound of the priors
        prior_high (jnp.array): Upper bound of the priors
        params_names (dict): Names of the parameters

    Returns:
        dict: Dictionary of key-value pairs of the parameters
    """
    params_dict = {}
    for low, high, param in zip(prior_low, prior_high, params_names):
        params_dict[param] = np.random.uniform(low, high)
    return params_dict

def generate_config(prior_low: np.array, 
                    prior_high: np.array, 
                    params_names: "list[str]", 
                    N_config: int = 1,
                    outdir: str = "./outdir/"
                    ) -> str:
    """
    From a given prior range and parameter names, generate the config files.
    
    Args:
        prior_low: lower bound of the prior range
        prior_high: upper bound of the prior range
        params_names: list of parameter names
        N_config: identification number of this config file.
    
    Returns:
        outdir (str): the directory where the config files are saved
    """
    
    # Generate parameters
    params_dict = generate_params_dict(prior_low, prior_high, params_names)
        
    # Create new injection file
    output_path = f'{outdir}injection_{str(N_config)}/'
    filename = output_path + f"config.json"
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Made injection directory: ", output_path)
    else:
        print("Injection directory exists: ", output_path)

    # This injection dictionary will store all needed information for the injection
    seed = np.random.randint(low=0, high=10000)
    injection_dict = {
        'seed': seed,
        'f_sampling': 2 * 2048,
        'duration': 128,
        'post_trigger_duration': 2,
        'trigger_time': 1187008882.43,
        'fmin': 20,
        'fref': 20,
        'ifos': ['H1', 'L1', 'V1'],
        'outdir' : output_path
    }
    
    # Combine these with the parameters
    injection_dict.update(params_dict)
    
    # Save the injection file to the output directory as JSON
    with open(filename, 'w') as f:
        json.dump(injection_dict, f)
    
    return injection_dict

def get_N(outdir):
    """
    Check outdir, get the subdirectories and return the length of subdirectories list.
    
    Useful to automatically generate the next injection directory without overriding other results.
    """
    subdirs = [x[0] for x in os.walk(outdir)]
    return len(subdirs)


def reflect_sky_location(
    gmst: Float,
    detectors: list[Detector],
    ra: Float,
    dec: Float,
    tc: Float,
    iota: Float
    ) -> tuple[Float, Float, Float]:

    assert len(detectors) == 3, "This reflection only holds for a 3-detector network"

    # convert tc to radian
    tc_rad = tc / (24 * 60 * 60) * 2 * jnp.pi

    # source location in cartesian coordinates
    # with the geocentric frame, thus the shift in ra by gmst
    v = jnp.array([jnp.cos(dec) * jnp.cos(ra - gmst - tc_rad),
                   jnp.cos(dec) * jnp.sin(ra - gmst - tc_rad),
                   jnp.sin(dec)])

    # construct the detector plane
    # fetch the detectors' locations
    x, y, z = detectors[0].vertex, detectors[1].vertex, detectors[2].vertex
    # vector normal to the detector plane
    n = jnp.cross(y - x, z - x)
    # normalize the vector
    nhat = n / jnp.sqrt(jnp.dot(n, n))
    # parametrize v as v = v_n * nhat + v_p, where nhat * v_p = 0
    v_n = jnp.dot(v, nhat)
    v_p = v - v_n * nhat
    # get the plan-reflect location
    v_ref = -v_n * nhat + v_p
    # convert back to ra, dec
    # adjust ra_prime so that it is in [0, 2pi)
    # i.e., negative values are map to [pi, 2pi)
    ra_prime = jnp.arctan2(v_ref[1], v_ref[0])
    ra_prime = ra_prime - (jnp.sign(ra_prime) - 1) * jnp.pi
    ra_prime = ra_prime + gmst + tc_rad  # add back the gps time and tc
    ra_prime = jnp.mod(ra_prime, 2 * jnp.pi)
    dec_prime = jnp.arcsin(v_ref[2])

    # calculate the time delay
    # just pick the first detector
    old_time_delay = detectors[0].delay_from_geocenter(ra, dec, gmst + tc_rad)
    new_time_delay = detectors[0].delay_from_geocenter(ra_prime, dec_prime,
                                                       gmst + tc_rad)
    tc_prime = tc + old_time_delay - new_time_delay
    
    # Also flip iota
    iota_prime = jnp.pi - iota

    return ra_prime, dec_prime, tc_prime, iota_prime

def generate_smart_initial_guess(gmst, 
                                 detectors, 
                                 ref_params, 
                                 n_chains, 
                                 n_dim,
                                 prior_low,
                                 prior_high,
                                 small_eps = 1e-3):
    """
    Generate a smart initial guess for the sky location and time of coalescence.
    
    NOTE: still under development. Now we are assuming fixed order of params (should be OK) and Uniform prior for the other parameters.
    """
    
    prior_ranges = prior_high - prior_low
    
    # Fetch the parameters
    mc, ra, dec, tc, iota = ref_params["M_c"], ref_params["ra"], ref_params["dec"], ref_params["t_c"], ref_params["iota"]
    
    ra_prime, dec_prime, tc_prime, iota_prime = reflect_sky_location(
        gmst, detectors, ra, dec, tc, iota
    )
    
    print("Going to initialize the walkers with the smart initial guess")

    # Fix the indices
    # TODO make these more generic, we are assuming fixed order of params for now
    mc_index = 0
    tc_index = 7
    iota_index = 9
    ra_index = 11
    dec_index = 12
    special_idx = [mc_index, tc_index, iota_index, ra_index, dec_index]
    other_idx = [i for i in range(n_dim) if i not in special_idx]
    
    # Standard deviations for the Gaussians
    mc_std = 0.1 # fix at 0.1 solar mass for now
    sky_std = 0.1 * jnp.array([prior_ranges[ra_index], prior_ranges[dec_index], prior_ranges[tc_index], prior_ranges[iota_index]])
    
    # Start new PRNG key
    my_seed = 123456
    my_key = jax.random.PRNGKey(my_seed)
    my_key, subkey = jax.random.split(my_key)
    
    # Chirp mass initialization
    z = jax.random.normal(subkey, shape = (int(n_chains),))
    
    mc_samples = mc + mc_std * z

    # Sample for ra, dec, tc, iota
    # TODO make less cumbersome here!
    # Prepare the samples, divided in half
    assert n_chains % 2 == 0, "n_chains must be multiple of two"
    n_chains_half = int(n_chains // 2) 
    my_key, subkey = jax.random.split(my_key)
    z = jax.random.normal(subkey, shape = (int(n_chains_half), 4))
    my_key, subkey = jax.random.split(my_key)
    z_prime = jax.random.normal(subkey, shape = (int(n_chains_half), 4))

    # True sky location
    ra, dec, tc, iota = ref_params["ra"], ref_params["dec"], ref_params["t_c"], ref_params["iota"]
    sky_means = jnp.array([ra, dec, tc, iota])
    sky_samples = sky_means + z * sky_std
    ra_samples, dec_samples, tc_samples, iota_samples = sky_samples[:,0], sky_samples[:,1], sky_samples[:,2], sky_samples[:,3]

    # Reflected sky location
    ra_prime, dec_prime, tc_prime, iota_prime = reflect_sky_location(gmst, detectors, ra, dec, tc, iota)
    sky_means_prime = jnp.array([ra_prime, dec_prime, tc_prime, iota_prime])
    sky_samples_prime = sky_means_prime + sky_std * z_prime
    ra_samples_prime, dec_samples_prime, tc_samples_prime, iota_samples_prime = sky_samples_prime[:,0], sky_samples_prime[:,1], sky_samples_prime[:,2], sky_samples_prime[:,3]

    # Merge original and reflected samples
    merged_ra   = jnp.concatenate([ra_samples, ra_samples_prime], axis=0)
    merged_dec  = jnp.concatenate([dec_samples, dec_samples_prime], axis=0)
    merged_tc   = jnp.concatenate([tc_samples, tc_samples_prime], axis=0)
    merged_iota = jnp.concatenate([iota_samples, iota_samples_prime], axis=0)

    # Rest of samples is uniform
    uniform_samples = jax.random.uniform(subkey, shape = (int(n_chains), n_dim - 5))
    for i, idx in enumerate(other_idx):
        # Get the relevant shift for this parameter, param is fetched by idx
        shift = prior_ranges[idx]
        # At this unifor/m samples, set the value using the shifts
        uniform_samples = uniform_samples.at[:,i].set(prior_low[idx] + uniform_samples[:,i] * shift)

    # Now build up the initial guess
    initial_guess = jnp.array([mc_samples, # Mc, 0
                               uniform_samples[:,0], # q, 1
                               uniform_samples[:,1], # chi1, 2
                               uniform_samples[:,2], # chi2, 3
                               uniform_samples[:,3], # lambda1, 4
                               uniform_samples[:,4], # lambda2, 5
                               uniform_samples[:,5], # dL, 6
                               merged_tc, # t_c, 7
                               uniform_samples[:,6], # phase_c, 8
                               jnp.cos(merged_iota), # cos_iota, 9
                               uniform_samples[:,7], # psi, 10
                               merged_ra, # ra, 11
                               jnp.sin(merged_dec), # sin_dec, 12
                            ]).T
    
    # Clip the initial guess to avoid the edges
    initial_guess = jnp.clip(initial_guess, prior_low + small_eps, prior_high - small_eps)
    
    return initial_guess

################
### PLOTTING ###
################

labels_results_plot = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda_1$', r'$\Lambda_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
            r'$t_c$', r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

labels_with_tc = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda$', r'$\delta\Lambda$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

def plot_accs(accs, label, name, outdir):
    
    eps = 1e-3
    plt.figure(figsize=(10, 6))
    plt.plot(accs, label=label)
    plt.ylim(0 - eps, 1 + eps)
    
    plt.ylabel(label)
    plt.xlabel("Iteration")
    plt.savefig(f"{outdir}{name}.png", bbox_inches='tight')  
    plt.close()
    
def plot_log_prob(log_prob, label, name, outdir):
    log_prob = np.mean(log_prob, axis = 0)
    plt.figure(figsize=(10, 6))
    plt.plot(log_prob, label=label)
    plt.yscale('log')
    plt.ylabel(label)
    plt.xlabel("Iteration")
    plt.savefig(f"{outdir}{name}.png", bbox_inches='tight')  
    plt.close()

    
def plot_chains(chains, name, outdir, truths = None, labels = labels_results_plot):
    
    chains = np.array(chains)
    
    # Check if 3D, then reshape
    if len(np.shape(chains)) == 3:
        chains = chains.reshape(-1, 13)
    
    # Find index of cos iota and sin dec
    cos_iota_index = labels.index(r'$\iota$')
    sin_dec_index = labels.index(r'$\delta$')
    # Convert cos iota and sin dec to cos and sin
    chains[:,cos_iota_index] = np.arccos(chains[:,cos_iota_index])
    chains[:,sin_dec_index] = np.arcsin(chains[:,sin_dec_index])
    chains = np.asarray(chains)
    fig = corner.corner(chains, labels = labels, truths = truths, hist_kwargs={'density': True}, **default_corner_kwargs)
    fig.savefig(f"{outdir}{name}.png", bbox_inches='tight')  
    
def plot_chains_from_file(outdir, load_true_params: bool = False):
    
    filename = outdir + 'results_production.npz'
    data = np.load(filename)
    chains = data['chains']
    my_chains = []
    n_dim = np.shape(chains)[-1]
    for i in range(n_dim):
        values = chains[:, :, i].flatten()
        my_chains.append(values)
    my_chains = np.array(my_chains).T
    chains = chains.reshape(-1, 13)
    if load_true_params:
        truths = load_true_params_from_config(outdir)
    else:
        truths = None
    
    plot_chains(chains, truths, 'results', outdir)
    
def plot_accs_from_file(outdir):
    
    filename = outdir + 'results_production.npz'
    data = np.load(filename)
    local_accs = data['local_accs']
    global_accs = data['global_accs']
    
    local_accs = np.mean(local_accs, axis = 0)
    global_accs = np.mean(global_accs, axis = 0)
    
    plot_accs(local_accs, 'local_accs', 'local_accs_production', outdir)
    plot_accs(global_accs, 'global_accs', 'global_accs_production', outdir)
    
def plot_log_prob_from_file(outdir, which_list = ['training', 'production']):
    
    for which in which_list:
        filename = outdir + f'results_{which}.npz'
        data = np.load(filename)
        log_prob= data['log_prob']
        plot_log_prob(log_prob, f'log_prob_{which}', f'log_prob_{which}', outdir)
    
    
def load_true_params_from_config(outdir):
    
    config = outdir + 'config.json'
    # Load the config   
    with open(config) as f:
        config = json.load(f)
    true_params = np.array([config[key] for key in NAMING])
    
    # Convert cos_iota and sin_dec to iota and dec
    cos_iota_index = NAMING.index('cos_iota')
    sin_dec_index = NAMING.index('sin_dec')
    true_params[cos_iota_index] = np.arccos(true_params[cos_iota_index])
    true_params[sin_dec_index] = np.arcsin(true_params[sin_dec_index])
    
    return true_params

def plot_loss_vals(loss_values, label, name, outdir):
    loss_values = loss_values.reshape(-1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label=label)
    
    plt.ylabel(label)
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.savefig(f"{outdir}{name}.png", bbox_inches='tight')  
    plt.close()

######################
### POSTPROCESSING ###
######################

def save_relative_binning_ref_params(likelihood: SingleEventLiklihood, outdir: str) -> None:
    """
    Save the relative binning and the reference parameters to a JSON file.

    Args:
        likelihood (SingleEventLiklihood): The likelihood object, must be HeterodynedTransientLikelihoodFD
        outdir (str): The output directory
    """
    if not isinstance(likelihood, HeterodynedTransientLikelihoodFD):
        print("This function is only compatible with HeterodynedTransientLikelihoodFD")
        return
    
    ref_params = likelihood.ref_params
    
    # Unpack to be compatible with JSON
    new_ref_params = {}
    for key, value in ref_params.items():
        # Check if value is an array or not, then convert to float
        if isinstance(value, Array):
            value = value.item()
        new_ref_params[key] = value
        
    # Save to JSON
    with open(f"{outdir}ref_params.json", 'w') as f:
        json.dump(new_ref_params, f)
        
def save_prior_bounds(prior_low: jnp.array, prior_high: jnp.array, outdir: str) -> None:
    """
    Save the prior bounds to a JSON file.

    Args:
        prior_low (jnp.array): Lower bound of the priors
        prior_high (jnp.array): Upper bound of the priors
        outdir (str): The output directory
    """
    
    my_dict = {}
    prior_low = prior_low.tolist()
    prior_high = prior_high.tolist()
    for (low, high), name in zip(zip(prior_low, prior_high), NAMING):
        my_dict[name] = list([low, high])
        
    with open(f"{outdir}prior_bounds.json", 'w') as f:
        json.dump(my_dict, f)

############
### MAIN ###
############

def main():
    # If wanted, can have a main here for postprocessing
    pass
    

if __name__ == "__main__":
    main()