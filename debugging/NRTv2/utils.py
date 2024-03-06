from astropy.time import Time
import jax.numpy as jnp
import jax
from jimgw.prior import Uniform, Composite
import numpy as np 
trigger_time = 1187008882.43
gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad

# Prior is copied over from the injection recovery script
PRIOR = {
        "M_c": [0.8759659737275101, 2.6060030916165484],
        "q": [0.5, 1.0], 
        "s1_z": [-0.05, 0.05], 
        "s2_z": [-0.05, 0.05], 
        "lambda_1": [0.0, 5000.0], 
        "lambda_2": [0.0, 5000.0], 
        "d_L": [30.0, 300.0], 
        "t_c": [-0.1, 0.1], 
        "phase_c": [0.0, 2 * np.pi], 
        "cos_iota": [-1.0, 1.0], 
        "psi": [0.0, np.pi], 
        "ra": [0.0, 2 * np.pi], 
        "sin_dec": [-1, 1]
}

naming = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']

bounds = np.array(list(PRIOR.values()))
prior_low = bounds[:, 0]
prior_high = bounds[:, 1]

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


def my_evaluate_vmap(likelihood, 
                     complete_prior,
                     samples, 
                     original = False):
    
    # Choose which evaluate to do
    if original:
        fn = likelihood.evaluate_original
    else:
        fn = likelihood.evaluate
        
    # Define the evaluate (with adding params etc) for single param sample
    def single_evaluate(sample):
        params = complete_prior.add_name(sample)
        params["gsmt"] = gmst
        params["dec"] = jnp.arcsin(params["sin_dec"])
        params["iota"] = jnp.arccos(params["cos_iota"])
        q = params["q"]
        params["eta"] = q / (1 + q) ** 2
        
        ln_L = fn(params, {})
        
        return ln_L
    
    # Now vmap that function
    result = jax.vmap(jax.jit(single_evaluate))(samples)
    
    return result
