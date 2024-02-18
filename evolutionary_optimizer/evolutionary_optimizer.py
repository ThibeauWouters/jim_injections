# The following is needed on CIT cluster to avoid an obscure Python error
import psutil
p = psutil.Process()
p.cpu_affinity([0])
# Regular imports 
import argparse
import os
import copy
import numpy as np
from astropy.time import Time
import time
import shutil
import json
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.single_event.waveform import RippleTaylorF2
from jimgw.prior import Uniform, Composite
import utils # our plotting and postprocessing utilities script

# Names of the parameters and their ranges for sampling parameters for the injection
NAMING = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']
PRIOR = {
        "M_c": [0.8759659737275101, 2.6060030916165484],
        "q": [0.5, 1.0], 
        "s1_z": [-0.05, 0.05], 
        "s2_z": [-0.05, 0.05], 
        "lambda_1": [0.0, 5000.0], 
        "lambda_2": [0.0, 5000.0], 
        "d_L": [30.0, 300.0], 
        "t_c": [-0.1, 0.1], 
        "phase_c": [0.0, 2 * jnp.pi], 
        "cos_iota": [-1.0, 1.0], 
        "psi": [0.0, jnp.pi], 
        "ra": [0.0, 2 * jnp.pi], 
        "sin_dec": [-1, 1]
}


################
### ARGPARSE ###
################

# TODO save these into a new file
def get_parser(**kwargs):
    add_help = kwargs.get("add_help", True)

    parser = argparse.ArgumentParser(
        description="Perform an injection recovery.",
        add_help=add_help,
    )
    parser.add_argument(
        "--GPU-device",
        type=int,
        default=0,
        help="Select GPU index to use.",
    )
    parser.add_argument(
        "--GPU-memory-fraction",
        type=float,
        default=0.5,
        help="Select percentage of GPU memory to use.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./outdir/",
        help="Output directory for the injection.",
    )
    parser.add_argument(
        "--load-existing-config",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to load and redo an existing injection (True) or to generate a new set of parameters (False).",
    )
    parser.add_argument(
        "--N",
        type=str,
        default="",
        help="Number (or generically, a custom identifier) of this injection, used to locate the output directory. If an empty string is passed (default), we generate a new injection.",
    )
    parser.add_argument(
        "--SNR-threshold",
        type=float,
        default=12,
        help="Skip injections with SNR below this threshold.",
    )
    parser.add_argument(
        "--waveform-approximant",
        type=str,
        default="TaylorF2",
        help="Which waveform approximant to use. Recommended to use TaylorF2 for now, NRTidalv2 might still be a bit unstable.",
    )
    parser.add_argument(
        "--relative-binning-binsize",
        type=int,
        default=100,
        help="Number of bins for the relative binning.",
    )
    parser.add_argument(
        "--relative-binning-ref-params-equal-true-params",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to set the reference parameters in the relative binning code to injection parameters.",
    )
    parser.add_argument(
        "--save-training-chains",
        type=bool,
        default=False,
        help="Whether to save training chains or not (can be very large!)",
    )
    parser.add_argument(
        "--eps-mass-matrix",
        type=float,
        default=1e-6,
        help="Overall scale factor to rescale the step size of the local sampler.",
    )
    parser.add_argument(
        "--smart-initial-guess",
        type=bool,
        default=False,
        help="Distribute the walkers around the injected parameters. TODO change this to reference parameters found by the relative binning code.",
    )
    # # TODO this has to be implemented
    # parser.add_argument(
    #     "--autotune_local_sampler",
    #     type=bool,
    #     default=False,
    #     help="TODO Still has to be implemented! Specify whether to use autotuning for the local sampler.",
    # )
    return parser
    
####################
### Script setup ###
####################

def body(args):
    """
    Run an injection and recovery. To get an explanation of the hyperparameters, go to:
        - jim hyperparameters: https://github.com/ThibeauWouters/jim/blob/8cb4ef09fefe9b353bfb89273a4bc0ee52060d72/src/jimgw/jim.py#L26
        - flowMC hyperparameters: https://github.com/ThibeauWouters/flowMC/blob/ad1a32dcb6984b2e178d7204a53d5da54b578073/src/flowMC/sampler/Sampler.py#L40
    """
    
    # TODO move and get these as arguments
    # Deal with the hyperparameters
    naming = NAMING
    HYPERPARAMETERS = {
    "flowmc": 
        {
            "n_loop_training": 400,
            "n_loop_production": 20,
            "n_local_steps": 5,
            "n_global_steps": 400,
            "n_epochs": 50,
            "n_chains": 1000, 
            "learning_rate": 0.001, 
            "max_samples": 50000, 
            "momentum": 0.9, 
            "batch_size": 50000, 
            "use_global": True, 
            "logging": True, 
            "keep_quantile": 0.5, 
            "local_autotune": None, 
            "train_thinning": 10, 
            "output_thinning": 30, 
            "n_sample_max": 10000, 
            "precompile": False, 
            "verbose": False, 
            "outdir": args.outdir
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
    
    flowmc_hyperparameters = HYPERPARAMETERS["flowmc"]
    jim_hyperparameters = HYPERPARAMETERS["jim"]
    hyperparameters = {**flowmc_hyperparameters, **jim_hyperparameters}
    
    for key, value in args.__dict__.items():
        if key in hyperparameters:
            hyperparameters[key] = value

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.GPU_memory_fraction)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU_device)
    print(f"Running on GPU {args.GPU_device}")

    print(f"Saving output to {args.outdir}")
    if args.waveform_approximant == "TaylorF2":
        ripple_waveform_fn = RippleTaylorF2
    else:
        raise ValueError("The waveform approximant is not recognized")

    # Before main code, check if outdir is correct dir format TODO improve with sys?
    if args.outdir[-1] != "/":
        args.outdir += "/"

    outdir = f"{args.outdir}injection_{args.N}/"
    # Save the given script hyperparams
    with open(f"{outdir}script_args.json", 'w') as json_file:
        json.dump(args.__dict__, json_file)
    
    # Get the prior bounds, both as 1D and 2D arrays
    prior_ranges = jnp.array([PRIOR[name] for name in naming])
    prior_low, prior_high = prior_ranges[:, 0], prior_ranges[:, 1]
    bounds = np.array(list(PRIOR.values()))
    
    # Now go over to creating parameters, and potentially check SNR cutoff
    network_snr = 0.0
    print(f"The SNR threshold parameter is set to {args.SNR_threshold}")
    while network_snr < args.SNR_threshold:
        # Generate the parameters or load them from an existing file
        if args.load_existing_config:
            config_path = f"{outdir}config.json"
            print(f"Loading existing config, path: {config_path}")
            config = json.load(open(config_path))
        else:
            print(f"Generating new config")
            config = utils.generate_config(prior_low, prior_high, naming, args.N, args.outdir)
        
        key = jax.random.PRNGKey(config["seed"])
        
        # Start injections
        print("Injecting signals . . .")
        waveform = ripple_waveform_fn(f_ref = config["fmin"])
        
        # Create frequency grid
        freqs = jnp.arange(
            config["fmin"],
            config["f_sampling"] / 2,  # maximum frequency being halved of sampling frequency
            1. / config["duration"]
            )
        # convert injected mass ratio to eta, and apply arccos and arcsin
        q = config["q"]
        eta = q / (1 + q) ** 2
        iota = float(jnp.arccos(config["cos_iota"]))
        dec = float(jnp.arcsin(config["sin_dec"]))
        # Setup the timing setting for the injection
        epoch = config["duration"] - config["post_trigger_duration"]
        gmst = Time(config["trigger_time"], format='gps').sidereal_time('apparent', 'greenwich').rad
        # Array of injection parameters
        true_param = {
            'M_c':       config["M_c"],       # chirp mass
            'eta':       eta,                 # symmetric mass ratio 0 < eta <= 0.25
            's1_z':      config["s1_z"],      # aligned spin of priminary component s1_z.
            's2_z':      config["s2_z"],      # aligned spin of secondary component s2_z.
            'lambda_1':  config["lambda_1"],  # tidal deformability of priminary component lambda_1.
            'lambda_2':  config["lambda_2"],  # tidal deformability of secondary component lambda_2.
            'd_L':       config["d_L"],       # luminosity distance
            't_c':       config["t_c"],       # timeshift w.r.t. trigger time
            'phase_c':   config["phase_c"],   # merging phase
            'iota':      iota,                # inclination angle
            'psi':       config["psi"],       # polarization angle
            'ra':        config["ra"],        # right ascension
            'dec':       dec                  # declination
            }
        
        # Get the true parameter values for the plots
        truths = copy.deepcopy(true_param)
        truths["eta"] = q
        truths = np.fromiter(truths.values(), dtype=float)
        
        detector_param = {
            'ra':     config["ra"],
            'dec':    dec,
            'gmst':   gmst,
            'psi':    config["psi"],
            'epoch':  epoch,
            't_c':    config["t_c"],
            }
        print(f"The injected parameters are {true_param}")
        
        # Generating the geocenter waveform
        h_sky = waveform(freqs, true_param)
        # Setup interferometers
        ifos = [H1, L1, V1]
        psd_files = ["./psds/psd.txt", "./psds/psd.txt", "./psds/psd_virgo.txt"]
        # inject signal into ifos
        for idx, ifo in enumerate(ifos):
            key, subkey = jax.random.split(key)
            ifo.inject_signal(
                subkey,
                freqs,
                h_sky,
                detector_param,
                psd_file=psd_files[idx]  # note: the function load_psd actaully loads the asd
            )
        print("Signal injected")
        
        # Compute the SNR
        h1_snr = utils.compute_snr(H1, h_sky, detector_param)
        l1_snr = utils.compute_snr(L1, h_sky, detector_param)
        v1_snr = utils.compute_snr(V1, h_sky, detector_param)
        network_snr = np.sqrt(h1_snr**2 + l1_snr**2 + v1_snr**2)
        
        # If the SNR is too low, we need to generate new parameters
        if network_snr < args.SNR_threshold:
            print(f"Network SNR is less than {args.SNR_threshold}, generating new parameters")
            if args.load_existing_config:
                raise ValueError("SNR is less than threshold, but loading existing config. This should not happen!")
    
    print("H1 SNR:", h1_snr)
    print("L1 SNR:", l1_snr)
    print("V1 SNR:", v1_snr)
    print("Network SNR:", network_snr)

    print("Start prior setup")
    
    # Priors without transformation 
    Mc_prior       = Uniform(prior_low[0], prior_high[0], naming=['M_c'])
    q_prior        = Uniform(prior_low[1], prior_high[1], naming=['q'],
                            transforms={
                                'q': (
                                    'eta',
                                    lambda params: params['q'] / (1 + params['q']) ** 2
                                    )
                                }
                            )
    s1z_prior      = Uniform(prior_low[2], prior_high[2], naming=['s1_z'])
    s2z_prior      = Uniform(prior_low[3], prior_high[3], naming=['s2_z'])
    lambda_1_prior = Uniform(prior_low[4], prior_high[4], naming=['lambda_1'])
    lambda_2_prior = Uniform(prior_low[5], prior_high[5], naming=['lambda_2'])
    dL_prior       = Uniform(prior_low[6], prior_high[6], naming=['d_L'])
    tc_prior       = Uniform(prior_low[7], prior_high[7], naming=['t_c'])
    phic_prior     = Uniform(prior_low[8], prior_high[8], naming=['phase_c'])
    cos_iota_prior = Uniform(prior_low[9], prior_high[9], naming=["cos_iota"],
                            transforms={
                                "cos_iota": (
                                    "iota",
                                    lambda params: jnp.arccos(
                                        jnp.arcsin(jnp.sin(params["cos_iota"] / 2 * jnp.pi)) * 2 / jnp.pi
                                    ),
                                )
                            },
                            )
    psi_prior      = Uniform(prior_low[10], prior_high[10], naming=["psi"])
    ra_prior       = Uniform(prior_low[11], prior_high[11], naming=["ra"])
    sin_dec_prior  = Uniform(prior_low[12], prior_high[12], naming=["sin_dec"],
        transforms={
            "sin_dec": (
                "dec",
                lambda params: jnp.arcsin(
                    jnp.arcsin(jnp.sin(params["sin_dec"] / 2 * jnp.pi)) * 2 / jnp.pi
                ),
            )
        },
    )
    
    # Save the prior bounds
    print("Saving prior bounds")
    utils.save_prior_bounds(prior_low, prior_high, outdir)
    
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

    print("Initializing likelihood")
    if args.relative_binning_ref_params_equal_true_params:
        ref_params = true_param
        print("Using the true parameters as reference parameters for the relative binning")
    else:
        ref_params = None
        print("Will search for reference waveform for relative binning")
        
    likelihood = HeterodynedTransientLikelihoodFD(
        ifos,
        prior=complete_prior,
        bounds=bounds,
        n_bins = args.relative_binning_binsize,
        waveform=waveform,
        trigger_time=config["trigger_time"],
        duration=config["duration"],
        post_trigger_duration=config["post_trigger_duration"],
        ref_params=ref_params # put the reference waveform of the relative binning at the true parameters
        )
    
    print("ref_params found:")
    print(likelihood.ref_params)
    
    # Save the ref params
    utils.save_relative_binning_ref_params(likelihood, outdir)

    print("Finished")

############
### MAIN ###
############

def main(given_args = None):
    
    parser = get_parser()
    args = parser.parse_args()
    
    print(given_args)
    
    # Update with given args
    if given_args is not None:
        args.__dict__.update(given_args)
        
    if args.load_existing_config and args.N == "":
        raise ValueError("If load_existing_config is True, you need to specify the N argument to locate the existing injection. ")
        
    print("------------------------------------")
    print("Arguments script:")
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
    print("------------------------------------")
        
    print("Starting main code")
    
    if len(args.N) == 0:
        N = utils.get_N(args.outdir)
        args.N = N
    
    start = time.time()
    body(args)
    end = time.time()
    print(f"Time taken: {end-start} seconds ({(end-start)/60} minutes)")
    
if __name__ == "__main__":
    main()