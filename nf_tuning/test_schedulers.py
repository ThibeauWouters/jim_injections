import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

params = {"axes.grid": True,
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

plt.rcParams.update(params)

import optax

n_epochs = 50
n_loop_training = 400
total_epochs = n_epochs * n_loop_training
start = int(total_epochs / 10)
start_lr = 1e-3
end_lr = 1e-6

# ### POLYNOMIAL SCHEDULER

# print("Original")
# power = 1.0
# schedule_fn = optax.polynomial_schedule(1e-3, 1e-4, 1.0, total_epochs-start, transition_begin=start)
# lrs_original = [schedule_fn(i) for i in range(total_epochs)]

# plt.figure(figsize=(10, 5))
# plt.plot(lrs_original)
# plt.xlabel('Epoch')
# plt.ylabel('Learning rate')
# plt.yscale('log')
# plt.title(f'Original LR schedule')
# plt.savefig('./figures/original_schedule.png')
# plt.show()
# plt.close()

# ### POLYNOMIAL SCHEDULER

# print("Polynomial")
# power = 4.0
# schedule_fn = optax.polynomial_schedule(start_lr, end_lr, power, total_epochs-start, transition_begin=start)
# lrs_polynomial = [schedule_fn(i) for i in range(total_epochs)]

# plt.figure(figsize=(10, 5))
# plt.plot(lrs_polynomial)
# plt.xlabel('Epoch')
# plt.ylabel('Learning rate')
# plt.yscale('log')
# plt.title(f'Polynomial: start {start_lr}, end {end_lr}, power {power}')
# plt.savefig('./figures/polynomial_schedule.png')
# plt.show()
# plt.close()

# ### EXPONENTIAL SCHEDULER

# print("Exponential")
# decay_rate = 0.5
# schedule_fn = optax.exponential_decay(start_lr, total_epochs-start, decay_rate, transition_begin=start, end_value=end_lr)
# lrs_exponential = [schedule_fn(i) for i in range(total_epochs)]

# plt.figure(figsize=(10, 5))
# plt.plot(lrs_exponential)
# plt.xlabel('Epoch')
# plt.ylabel('Learning rate')
# plt.yscale('log')
# plt.title(f'Exponential: start {start_lr}, end {end_lr}, rate {decay_rate}')
# plt.savefig('./figures/exponential_schedule.png')
# plt.show()
# plt.close()

# ### COMBINED

# print("Combined")
# plt.figure(figsize=(10, 5))
# plt.plot(lrs_polynomial)
# plt.plot(lrs_exponential)
# plt.xlabel('Epoch')
# plt.ylabel('Learning rate')
# plt.yscale('log')
# plt.title(f'Combined')
# # plt.savefig('./figures/combined.png')
# plt.show()
# plt.close()


####### AUTOTUNE PROPOSALS, BASED ON GLOBAL ACCEPTANCE RATE #######

def compute_learning_rate(acc, start_lr: float = 1e-3, end_lr: float = 1e-7, power: float = 4.0):
    """ 
    Polynomial learning rate schedule.
    """
    learning_rate = (start_lr - end_lr) * (1 - acc) ** power + end_lr
    return learning_rate

acc_array = np.linspace(0, 1, 1_000)
lr_array = np.array([compute_learning_rate(acc) for acc in acc_array])

print("My autotune proposal")
plt.figure(figsize=(10, 5))
plt.plot(acc_array, lr_array)
plt.xlabel('Acceptance')
plt.ylabel('Learning rate')
plt.yscale('log')
plt.title(f'Autotune proposal', fontsize=16)
plt.savefig('./figures/autotune_proposal.png')
plt.show()
plt.close()