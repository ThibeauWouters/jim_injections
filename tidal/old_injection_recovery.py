import os
### Performance GPU flags -- not working?
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_enable_triton_softmax_fusion=true '
#     '--xla_gpu_triton_gemm_any=True '
#     '--xla_gpu_enable_async_collectives=true '
#     '--xla_gpu_enable_latency_hiding_scheduler=true '
#     '--xla_gpu_enable_highest_priority_async_stream=true '
# )
# os.environ.update({
#   "NCCL_LL128_BUFFSIZE": "-2",
#   "NCCL_LL_BUFFSIZE": "-2",
#    "NCCL_PROTO": "SIMPLE,LL,LL128",
#  })
# TODO remove, this is to check memory usage
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import shutil
import sys
import json
from scipy.interpolate import interp1d
# jim
from jimgw.jim import Jim
from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1, V1, Detector
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.single_event.waveform import RippleTaylorF2
from jimgw.prior import Uniform, Composite
# jax
import jax.numpy as jnp
import jax
import jax.profiler
# ripple
from ripple import Mc_eta_to_ms
# others
import numpy as np
jax.config.update("jax_enable_x64", True)
from astropy.time import Time

import numpy as np
import corner
import matplotlib.pyplot as plt

import time

# import lalsimulation as lalsim

import gc

####################
### Script setup ###
####################

### Script constants
SNR_THRESHOLD = 0.001 # skip injections with SNR below this threshold
override_PSD = False # whether to load another PSD file -- unused now
use_lambda_tildes = False # whether to use lambda tildes instead of individual component lambdas or not
duration_with_lalsim = False # TODO check with Peter whether this is OK/useful?
waveform_approximant = "TaylorF2" # which waveform approximant to use, either TaylorF2 or IMRPhenomD_NRTidalv2
print(f"Waveform approximant: {waveform_approximant}")
OUTDIR = f"./outdir/"
# load_existing_config = True # whether to load an existing config file or generate a new one on the fly

### Script hyperparameters

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
                        save=False)

matplotlib_params = {
    "axes.labelsize": 30,
    "axes.titlesize": 30,
    "text.usetex": True,
    "font.family": "serif",
}
plt.rcParams.update(matplotlib_params)

# These are the labels that we use when plotting right after an injection has finished
if use_lambda_tildes:
    labels_results_plot = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\tilde{\Lambda}$', r'$\delta\tilde{\Lambda}$', r'$d_{\rm{L}}/{\rm Mpc}$',
                r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']
else:
    labels_results_plot = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda_1$', r'$\Lambda_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
                r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

### Shared injection constants

HYPERPARAMETERS = {
    "flowmc": 
        {
            "n_loop_training": 130, # 130 
            "n_loop_production": 20, # 50
            "n_local_steps": 200,
            "n_global_steps": 200, # 200
            "n_chains": 1000, 
            "n_epochs": 100, # 100
            "learning_rate": 0.001, 
            "max_samples": 50000, 
            "momentum": 0.9, 
            "batch_size": 50000, 
            "use_global": True, 
            "logging": True, 
            "keep_quantile": 0.0, 
            "local_autotune": None, 
            "train_thinning": 10, 
            "output_thinning": 30, 
            "n_sample_max": 10000, 
            "precompile": False, 
            "verbose": False, 
            "outdir_name": OUTDIR
        }, 
    "jim": 
        {
            "seed": 0, 
            "n_chains": 1000, 
            "num_layers": 10, 
            "hidden_size": [128, 128], 
            "num_bins": 8, 
        }
}

### Backing up the original hyperparameters to be sure
# HYPERPARAMETERS = {
#     "flowmc": 
#         {
#             "n_loop_training": 150, 
#             "n_loop_production": 50, 
#             "n_local_steps": 200,
#             "n_global_steps": 200, 
#             "n_chains": 1000, 
#             "n_epochs": 50, 
#             "learning_rate": 0.001, 
#             "max_samples": 50000, 
#             "momentum": 0.9, 
#             "batch_size": 50000, 
#             "use_global": True, 
#             "logging": True, 
#             "keep_quantile": 0.0, 
#             "local_autotune": None, 
#             "train_thinning": 10, 
#             "output_thinning": 30, 
#             "n_sample_max": 10000, 
#             "precompile": False, 
#             "verbose": False, 
#             "outdir_name": OUTDIR
#         }, 
#     "jim": 
#         {
#             "seed": 0, 
#             "n_chains": 1000, 
#             "num_layers": 10, 
#             "hidden_size": [128, 128], 
#             "num_bins": 8, 
#         }
# }

# For Mc prior: 0.870551 if lower bound is 1, 0.435275 if lower bound is 0.5
MC_PRIOR_1 = [0.8759659737275101, 2.6060030916165484] # lowest individual mass is 1
# MC_LOW_PRIOR = [0.435275, 0.870551] # NOTE only for testing low Mc values, arbitrarily chose to be between above 2 lower bounds

if use_lambda_tildes:
    NAMING = ['M_c', 'q', 's1_z', 's2_z', 'lambda_tilde', 'delta_lambda_tilde', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']
    PRIOR = {
        "M_c": MC_PRIOR_1,
        "q": [0.5, 1.0], 
        "s1_z": [-0.05, 0.05], 
        "s2_z": [-0.05, 0.05], 
        "lambda_tilde": [0.0, 9000.0], 
        "delta_lambda_tilde": [-1000.0, 1000.0], 
        "d_L": [30.0, 300.0], 
        "t_c": [-0.1, 0.1], 
        "phase_c": [0.0, 6.283185307179586], 
        "cos_iota": [-1.0, 1.0], 
        "psi": [0.0, 3.141592653589793], 
        "ra": [0.0, 6.283185307179586], 
        "sin_dec": [-1, 1]
    }
else:
        NAMING = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']
        PRIOR = {
        "M_c": MC_PRIOR_1,
        "q": [0.5, 1.0], 
        "s1_z": [-0.05, 0.05], 
        "s2_z": [-0.05, 0.05], 
        "lambda_1": [0.0, 5000.0], 
        "lambda_2": [0.0, 5000.0], 
        "d_L": [30.0, 300.0], 
        "t_c": [-0.1, 0.1], 
        "phase_c": [0.0, 6.283185307179586], 
        "cos_iota": [-1.0, 1.0], 
        "psi": [0.0, 3.141592653589793], 
        "ra": [0.0, 6.283185307179586], 
        "sin_dec": [-1, 1]
    }



#########################
### Utility functions ###
#########################

def compute_snr(detector, h_sky, detector_params):
    """Compute the SNR of an event for a single detector, given the waveform generated in the sky.

    Args:
        detector (Detector): Detector object from jim.
        h_sky (Array): Jax numpy array containing the waveform strain as a function of frequency in the sky frame
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

def generate_params_dict(prior_low, prior_high, params_names):
    # First, sample uniformly
    params_dict = {}
    for low, high, param in zip(prior_low, prior_high, params_names):
        params_dict[param] = np.random.uniform(low, high)
    
    ### NOTE - powerlaw prior here
    # # Then, for distance, sample with the powerlaw
    # xmin_distance = 30.0
    # xmax_distance = 300.0
    # dL_prior = Powerlaw(xmin_distance, xmax_distance, alpha = 2.0, naming=["d_L"])
    
    # # Generate the dL sample from a Powerlaw prior instead of a uniform prior
    # seed = np.random.randint(low=0, high=10000)
    # key, subkey = jax.random.split(jax.random.PRNGKey(seed + 42))
    # params_dict["d_L"] = dL_prior.sample(subkey, 1)["d_L"].item()
    
    return params_dict

def generate_config(prior_low: np.array, 
                    prior_high: np.array, 
                    params_names: "list[str]", 
                    N_config: int = 1,
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
    
    params_dict = generate_params_dict(prior_low, prior_high, params_names)
        
    # Create new injection file
    output_path = f'{OUTDIR}injection_{str(N_config)}/'
    filename = output_path + f"config.json"
    
    # Check if directory exists, if not, create it. Otherwise, delete it and create it again
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    print("Made injection directory")

    # This injection dictionary will store all needed information for the injection
    seed = np.random.randint(low=0, high=10000)
    injection_dict = {
        'seed': seed,
        'f_sampling': 2*2048,
        'duration': 256,
        'fmin': 20,
        'ifos': ['H1', 'L1', 'V1'],
        'outdir' : output_path
    }
    
    injection_dict.update(params_dict)
    
    # Save the injection file to the output directory as JSON
    with open(filename, 'w') as f:
        json.dump(injection_dict, f)
    
    return injection_dict

def get_N(outdir):
    """
    Check outdir, get the subdirectories and return the length of subdirectories list
    """
    subdirs = [x[0] for x in os.walk(outdir)]
    return len(subdirs)


###############################
### Main body of the script ###
###############################

def interp_psd(freqs, f_psd, psd_vals):
    psd = interp1d(f_psd, psd_vals, fill_value=(psd_vals[0], psd_vals[-1]))(freqs)
    return psd

def body(N, outdir, load_existing_config = False):

    network_snr = 0.0
    print(f"The SNR threshold parameter is set to {SNR_THRESHOLD}")

    # Get naming and prior bounds
    while network_snr < SNR_THRESHOLD:
        naming = list(PRIOR.keys())
        prior_ranges = jnp.array([PRIOR[name] for name in naming])
        prior_low, prior_high = prior_ranges[:, 0], prior_ranges[:, 1]

        if load_existing_config:
            print("Loading existing config, path:")
            config_path = f"{outdir}injection_{str(N)}/config.json"
            config = json.load(open(config_path))
        else:
            config = generate_config(prior_low, prior_high, naming, N)
        outdir = config["outdir"]

        print(PRIOR.items())

        naming = list(PRIOR.keys())
        bounds = []
        for key, value in PRIOR.items():
            bounds.append(value)

        bounds = np.asarray(bounds)
        xmin = bounds[:, 0]
        xmax = bounds[:, 1]

        # Fetch the flowMC and jim hyperparameters, and put together into one dict:
        flowmc_hyperparameters = HYPERPARAMETERS["flowmc"]
        jim_hyperparameters = HYPERPARAMETERS["jim"]
        hyperparameters = {**flowmc_hyperparameters, **jim_hyperparameters}

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
        print(f"Sampling from prior (for tidal parameters: use_lambda_tildes = {use_lambda_tildes})")
        
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
        
        if use_lambda_tildes:
            first_lambda_prior  = Uniform(xmin[4], xmax[4], naming=["lambda_tilde"])
            second_lambda_prior = Uniform(xmin[5], xmax[5], naming=["delta_lambda_tilde"])
        else:
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
        print("true_params")
        print(true_params)
        
        ## TODO check this Compute the duration of the signal with lalsim
        # if duration_with_lalsim:
        #     mc = true_params["M_c"]
        #     eta = true_params["q"] / (1 + true_params["q"]) ** 2
        #     s1 = true_params["s1_z"]
        #     s2 = true_params["s2_z"]
        #     fstart = fmin
            
        #     m1, m2 = Mc_eta_to_ms(jnp.array([mc, eta])) # in solar masses
        #     m1, m2 = m1 * lalsim.lal.MSUN_SI, m2 * lalsim.lal.MSUN_SI # in kg
        #     tchirp = lalsim.SimInspiralChirpTimeBound(fstart, m1, m2, s1, s2)
        #     T = tchirp
        #     duration = T
        # else:
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
        
        ## TODO what PSD is this?
        
        print("Injecting signal. No noise is set to: ")
        key, subkey = jax.random.split(jax.random.PRNGKey(config["seed"] + 1234))
        H1.inject_signal(subkey, freqs, h_sky, detector_param, psd_file = "psd.txt")
        key, subkey = jax.random.split(key)
        L1.inject_signal(subkey, freqs, h_sky, detector_param, psd_file = "psd.txt")
        key, subkey = jax.random.split(key)
        V1.inject_signal(subkey, freqs, h_sky, detector_param, psd_file = "psd_virgo.txt")
        
        # if override_PSD:
        #     H1_psd_frequency, H1_psd = np.genfromtxt(f'{jim_data_dir}GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_H1_psd.txt').T
        #     H1_psd_interp = interp_psd(freqs, H1_psd_frequency, H1_psd)
            
        #     L1_psd_frequency, L1_psd = np.genfromtxt(f'{jim_data_dir}GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_L1_psd.txt').T
        #     L1_psd_interp = interp_psd(freqs, L1_psd_frequency, L1_psd)
            
        #     V1_psd_frequency, V1_psd = np.genfromtxt(f'{jim_data_dir}GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_V1_psd.txt').T
        #     V1_psd_interp = interp_psd(freqs, V1_psd_frequency, V1_psd)
        #     print("We override the PSD with the results from the GW170817 files")
        #     H1.psd = H1_psd_interp
        #     L1.psd = L1_psd_interp
        #     V1.psd = V1_psd_interp
        #     print("We override the PSD with the results from the GW170817 files - DONE")
        
        # Compute the SNR and add it to a file for use later on
        
        h1_snr = compute_snr(H1, h_sky, detector_param)
        l1_snr = compute_snr(L1, h_sky, detector_param)
        v1_snr = compute_snr(V1, h_sky, detector_param)
        
        network_snr = np.sqrt(h1_snr**2 + l1_snr**2 + v1_snr**2)
        
        print("Network SNR:", network_snr)
        if network_snr < SNR_THRESHOLD:
            print(f"Network SNR is less than {SNR_THRESHOLD}, generating new parameters")
    
    print("Parameters accepted (above SNR threshold))")
    string_snr = f"SNR of injection:\nH1: {h1_snr}\nL1: {l1_snr}\nV1: {v1_snr}\nNetwork: {network_snr}"
    print(string_snr)
    
    # Now, in the given outdir, write these to a file
    snr_file = outdir + "snr.csv"
    snr_dict = {"detector": ["H1", "L1", "V1", "network"], 
                "snr": [h1_snr, l1_snr, v1_snr, network_snr]}
    
    
    # Also write string_snr to a txt file for easy access
    snr_txt_file = outdir + "snr.txt"
    with open(snr_txt_file, "w") as f:
        f.write(string_snr)
    
    ### Create likelihood object

    # print("Creating likelihood object")
    # if waveform_approximant == "IMRPhenomD_NRTidalv2":
    #     waveform = RippleIMRPhenomD_NRTidalv2(use_lambda_tildes=use_lambda_tildes)
    
    waveform = RippleTaylorF2(use_lambda_tildes=use_lambda_tildes)
        
    nb_bins = 100
    likelihood = HeterodynedTransientLikelihoodFD([H1, L1, V1], prior=prior, bounds=bounds, waveform=waveform, trigger_time=gps, duration=T, n_bins=nb_bins, ref_params=true_params)
    print("Creating likelihood object: done")

    ### Create sampler and jim objects
    # if waveform_approximant == "IMRPhenomD_NRTidalv2":
    #     eps = 1e-2 # TODO test if this can be changed?
    # else:
    eps = 1e-2
    
    n_dim = 13
    mass_matrix = jnp.eye(n_dim)
    mass_matrix = mass_matrix.at[0,0].set(1e-5)
    mass_matrix = mass_matrix.at[1,1].set(1e-4)
    mass_matrix = mass_matrix.at[2,2].set(1e-3)
    mass_matrix = mass_matrix.at[3,3].set(1e-3)
    mass_matrix = mass_matrix.at[7,7].set(1e-5)
    mass_matrix = mass_matrix.at[11,11].set(1e-2)
    mass_matrix = mass_matrix.at[12,12].set(1e-2)
    mass_matrix = eps * mass_matrix
    
    local_sampler_arg = {"step_size": mass_matrix}
    hyperparameters["outdir_name"] = outdir
    hyperparameters["local_sampler_arg"] = local_sampler_arg

    jim = Jim(
        likelihood,
        prior,
        **hyperparameters
    )

    ### Heavy computation begins
    jim.sample(jax.random.PRNGKey(42))
    ### Heavy computation ends

    # === Show results, save output ===

    ### Summary to screen:
    jim.print_summary()
    
    print("For comparison, the true parameters were:")
    print(true_params)

    ### Diagnosis plots of summaries
    print("Saving chains")
    # jim.Sampler.plot_summary("training")
    # jim.Sampler.plot_summary("production")

    ### Save samples for the production stage (training samples is too expensive for memory)
    name = outdir + f'results_production.npz'
    print(f"Saving production samples in npz format to {name}")
    state = jim.Sampler.get_sampler_state("production")
    chains, log_prob, local_accs, global_accs = state["chains"], state["log_prob"], state["local_accs"], state["global_accs"]
    np.savez(name, chains=chains, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs)

    print("Sampling from the flow")
    chains = jim.Sampler.sample_flow(10000)
    name = outdir + 'results_NF.npz'
    print(f"Saving flow samples to {name}")
    np.savez(name, chains=chains)

    ### Plot chains and samples
    

    # Production samples:
    file = outdir + "results_production.npz"
    name = outdir + "results_production.png"

    data = np.load(file)
    # Ignore t_c, and reshape with n_dims, and do conversions
    idx_list = [0,1,2,3,4,5,6,8,9,10,11,12]
    chains = data['chains'][:,:,idx_list].reshape(-1,12)
    chains[:,8] = np.arccos(chains[:,8])
    chains[:,11] = np.arcsin(chains[:,11])
    chains = np.asarray(chains)
    fig = corner.corner(chains, labels = labels_results_plot, hist_kwargs={'density': True}, **default_corner_kwargs)
    fig.savefig(name, bbox_inches='tight')  

    # NF samples:
    file = outdir + "results_NF.npz"
    name = outdir + "results_NF.png"

    data = np.load(file)["chains"]

    chains = data[:, idx_list]
    chains = np.asarray(chains)
    fig = corner.corner(chains, labels = labels_results_plot, hist_kwargs={'density': True}, **default_corner_kwargs)
    fig.savefig(name, bbox_inches='tight')  
        
    print("Saving the jim hyperparameters")
    jim.save_hyperparameters()
    
    print("Saving the NF")
    jim.Sampler.save_flow(outdir + "nf_model")

    
    # Remove in order to hopefully reduce memory?
    jim = None
    gc.collect()
    print("INJECTION RECOVERY FINISHED SUCCESSFULLY")
    
def main():
    
    # ## Normal, new injection:
    # N = get_N(OUTDIR)
    # my_string = "================================================================================================================================================================================================================================================"
    # print(my_string)
    # print(f"Running injection script for  N = {N}")
    # print(my_string)
    # body(N, outdir=OUTDIR) # regular, computing on the fly
    
    # ### Rerun a specific injection
    body(7, outdir = OUTDIR, load_existing_config = True) 
    
    # jax.profiler.save_device_memory_profile("memory.prof")
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {end-start} seconds ({(end-start)/60} minutes)")
    
