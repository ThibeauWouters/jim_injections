import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import optax

n_epochs = 50
n_loop_training = 400
total_epochs = n_epochs * n_loop_training
start = int(total_epochs / 10)
start_lr = 1e-3
end_lr = 1e-4
power = 2.0
schedule_fn = optax.polynomial_schedule(start_lr, end_lr, power, total_epochs-start, transition_begin=start)

lrs = [schedule_fn(i) for i in range(total_epochs)]

plt.figure(figsize=(10, 5))
plt.plot(lrs)
plt.xlabel('Epoch')
plt.ylabel('Learning rate')
plt.title(f'Polynomial: start {start_lr}, end {end_lr}, power {power}')
plt.savefig('./figures/polynomial_schedule.png')
plt.show()
