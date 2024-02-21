import optax 
import equinox as eqx
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.utils.PRNG_keys import initialize_rng_keys

rng_key_set = initialize_rng_keys(1000, seed=42)
nf_model = MaskedCouplingRQSpline(
            13, 10, [128,128], 8, rng_key_set[-1]
        )
learning_rate = 1e-3
tx = optax.chain(optax.clip(1.0), optax.adam(learning_rate))
optim_state = tx.init(eqx.filter(nf_model, eqx.is_array))
print("tx")
print(tx)
print("tx.update")
print(tx.update)
# print("optim_state")
# print(optim_state)