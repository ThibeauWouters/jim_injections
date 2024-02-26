import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"
import psutil
p = psutil.Process()
p.cpu_affinity([0])

import optax

from jimgw.prior import Uniform, Composite
from jimgw.single_event.waveform import RippleTaylorF2, RippleIMRPhenomD_NRTidalv2
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.single_event.detector import H1, L1, V1
from jimgw.jim import Jim
import jax.numpy as jnp
import jax
import json
import shutil
import time
from astropy.time import Time
import argparse
import numpy as np
import copy

jax.config.update("jax_enable_x64", True)

import sys
sys.path.append("../tidal/")
import utils  # our plotting and postprocessing utilities script

# Names of the parameters and their ranges for sampling parameters for the injection
NAMING = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2',
          'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']
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


####################
### Script setup ###
####################

def body(args):
    """
    Run an injection and recovery. To get an explanation of the hyperparameters, go to:
        - jim hyperparameters: https://github.com/ThibeauWouters/jim/blob/8cb4ef09fefe9b353bfb89273a4bc0ee52060d72/src/jimgw/jim.py#L26
        - flowMC hyperparameters: https://github.com/ThibeauWouters/flowMC/blob/ad1a32dcb6984b2e178d7204a53d5da54b578073/src/flowMC/sampler/Sampler.py#L40
    """

    start_time = time.time()
    naming = NAMING

    # POLYNOMIAL SCHEDULER
    if args.use_scheduler:
        print("Using polynomial learning rate scheduler")
        total_epochs = args.n_epochs * args.n_loop_training
        start = int(total_epochs / 10)
        start_lr = 1e-3
        end_lr = 1e-5
        power = 4.0
        schedule_fn = optax.polynomial_schedule(
            start_lr, end_lr, power, total_epochs-start, transition_begin=start)
        args.learning_rate = schedule_fn
        scheduler_string = f"Polynomial scheduler: start_lr = {start_lr}, end_lr = {end_lr}, power = {power}, start = {start}"

    print(f"Saving output to {args.outdir}")

    # Fetch waveform used
    supported_waveforms = ["TaylorF2", "NRTidalv2", "IMRPhenomD_NRTidalv2"]
    if args.waveform_approximant not in supported_waveforms:
        print(
            f"Waveform approximant {args.waveform_approximant} not supported. Supported waveforms are {supported_waveforms}. Changing to TaylorF2.")
        args.waveform_approximant = "TaylorF2"

    if args.waveform_approximant == "TaylorF2":
        ripple_waveform_fn = RippleTaylorF2
    elif args.waveform_approximant == "IMRPhenomD_NRTidalv2":
        ripple_waveform_fn = RippleIMRPhenomD_NRTidalv2
        # NOTE we are going to disable the local sampler for now when using NRTv2 (possible bug in jax.grad?)
        args.use_local = False
    else:
        raise ValueError(
            f"Something went wrong with initializing waveform approximant function.")

    # Before main code, check if outdir is correct dir format TODO improve with sys?
    if args.outdir[-1] != "/":
        args.outdir += "/"

    outdir = f"{args.outdir}injection_{args.N}/"

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
            config = utils.generate_config(
                prior_low, prior_high, naming, args.N, args.outdir)

        key = jax.random.PRNGKey(config["seed"])

        # Save the given script hyperparams
        with open(f"{outdir}script_args.json", 'w') as json_file:
            # TODO make this less hacky
            if args.use_scheduler:
                my_dict = copy.deepcopy(args.__dict__)
                my_dict["learning_rate"] = scheduler_string
            json.dump(my_dict, json_file)

        # Start injections
        print("Injecting signals . . .")
        waveform = ripple_waveform_fn(f_ref=config["fref"])

        # Create frequency grid
        freqs = jnp.arange(
            config["fmin"],
            # maximum frequency being halved of sampling frequency
            config["f_sampling"] / 2,
            1. / config["duration"]
        )
        # convert injected mass ratio to eta, and apply arccos and arcsin
        q = config["q"]
        eta = q / (1 + q) ** 2
        iota = float(jnp.arccos(config["cos_iota"]))
        dec = float(jnp.arcsin(config["sin_dec"]))
        # Setup the timing setting for the injection
        epoch = config["duration"] - config["post_trigger_duration"]
        gmst = Time(config["trigger_time"], format='gps').sidereal_time(
            'apparent', 'greenwich').rad
        # Array of injection parameters
        true_param = {
            'M_c':       config["M_c"],       # chirp mass
            'eta':       eta,                 # symmetric mass ratio 0 < eta <= 0.25
            # aligned spin of priminary component s1_z.
            's1_z':      config["s1_z"],
            # aligned spin of secondary component s2_z.
            's2_z':      config["s2_z"],
            # tidal deformability of priminary component lambda_1.
            'lambda_1':  config["lambda_1"],
            # tidal deformability of secondary component lambda_2.
            'lambda_2':  config["lambda_2"],
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
        psd_files = ["./psds/psd.txt",
                     "./psds/psd.txt", "./psds/psd_virgo.txt"]
        # inject signal into ifos
        for idx, ifo in enumerate(ifos):
            key, subkey = jax.random.split(key)
            ifo.inject_signal(
                subkey,
                freqs,
                h_sky,
                detector_param,
                # note: the function load_psd actaully loads the asd
                psd_file=psd_files[idx]
            )
        print("Signal injected")

        # Compute the SNR
        h1_snr = utils.compute_snr(H1, h_sky, detector_param)
        l1_snr = utils.compute_snr(L1, h_sky, detector_param)
        v1_snr = utils.compute_snr(V1, h_sky, detector_param)
        network_snr = np.sqrt(h1_snr**2 + l1_snr**2 + v1_snr**2)

        # If the SNR is too low, we need to generate new parameters
        if network_snr < args.SNR_threshold:
            print(
                f"Network SNR is less than {args.SNR_threshold}, generating new parameters")
            if args.load_existing_config:
                raise ValueError(
                    "SNR is less than threshold, but loading existing config. This should not happen!")

    print("H1 SNR:", h1_snr)
    print("L1 SNR:", l1_snr)
    print("V1 SNR:", v1_snr)
    print("Network SNR:", network_snr)

    print(f"Saving network SNR")
    with open(outdir + 'network_snr.txt', 'w') as file:
        file.write(str(network_snr))
        
    for name, ifo in zip(["H1", "L1", "V1"], ifos):
        print(f"Saving injected signal as Numpy arrays for {name}")
        np.savez(f"{outdir}{name}_data.npz", freqs=ifo.frequencies, data=ifo.data)

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

    ref_params = [1.95713842e+00,
     9.10050622e-01,
     3.40685893e-02,
     2.72165114e-02,
     4.94829270e+02,
     4.37424050e+03,
     8.21636616e+01,
     -2.30779116e-02,
     5.97207208e+00, 
     -5.23343085e-01,
     3.02950901e+00,
     5.98376982e+00,
     1.28732911e-03]
    
    # Print difference with the true params
    diffs = abs(ref_params - truths)
    
    ref_params = jnp.array(ref_params)
    ref_params = complete_prior.add_name(ref_params)
    
    ref_params["iota"] = jnp.arccos(ref_params["cos_iota"])
    ref_params["dec"] = jnp.arcsin(ref_params["sin_dec"])
    q = ref_params["q"]
    ref_params["eta"] = q / (1 + q) ** 2

    likelihood = HeterodynedTransientLikelihoodFD(
        ifos,
        prior=complete_prior,
        bounds=bounds,
        n_bins=args.relative_binning_binsize,
        waveform=waveform,
        trigger_time=config["trigger_time"],
        duration=config["duration"],
        post_trigger_duration=config["post_trigger_duration"],
        # put the reference waveform of the relative binning at the true parameters
        ref_params=true_param
    )

    # Save the ref params
    utils.save_relative_binning_ref_params(likelihood, outdir)
    
    # Now try to save it
    
    try:
        import pickle
        with open(f"{outdir}likelihood.pkl", 'wb') as file:
            pickle.dump(likelihood, file)
    except Exception as e:
        print(f"Could not save likelihood: \n {e}")

    print("Finished successfully!")

############
### MAIN ###
############


def main(given_args=None):

    # Get the parser
    parser = utils.get_parser()
    args = parser.parse_args()

    # Update with given args
    if given_args is not None:
        args.__dict__.update(given_args)

    if args.load_existing_config and args.N == "":
        raise ValueError(
            "If load_existing_config is True, you need to specify the N argument to locate the existing injection. ")

    print("------------------------------------------------")
    print("Arguments script:")
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
    print("------------------------------------------------")

    print("Starting main code")

    # If no N is given, fetch N from the structure of outdir
    if len(args.N) == 0:
        N = utils.get_N(args.outdir)
        args.N = N

    # Execute the script
    body(args)


if __name__ == "__main__":
    main()
