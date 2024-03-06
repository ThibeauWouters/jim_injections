"""
Deprecated
"""


# import warnings
# warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
# import psutil
# p = psutil.Process()
# p.cpu_affinity([0])
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import jax
# jax.config.update("jax_enable_x64", True)
# print(jax.devices())

# import utils

# params = {"axes.grid": True,
#         "text.usetex" : True,
#         "font.family" : "serif",
#         "ytick.color" : "black",
#         "xtick.color" : "black",
#         "axes.labelcolor" : "black",
#         "axes.edgecolor" : "black",
#         "font.serif" : ["Computer Modern Serif"],
#         "xtick.labelsize": 16,
#         "ytick.labelsize": 16,
#         "axes.labelsize": 16,
#         "legend.fontsize": 16,
#         "legend.title_fontsize": 16,
#         "figure.titlesize": 16}

# plt.rcParams.update(params)
# naming = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']

# def plot_sampled_likelihoods(idx1_list: list, idx2_list: list):
#     filename = "evaluated_likelihoods.npz"

#     data = np.load(filename)
#     samples = data["samples"]
#     log_likelihood = data["log_likelihood"]
#     log_likelihood_original = data["log_likelihood_original"]

#     for (idx1, idx2) in zip(idx1_list, idx2_list):

#         name1 = naming[idx1]
#         name2 = naming[idx2]

#         title_names = ["log likelihood", "log likelihood original"]

#         for name, values in zip(title_names, [log_likelihood, log_likelihood_original]):
#             fig, ax = plt.subplots(1, 1, figsize = (8, 8))
#             sc = ax.scatter(samples[:, idx1], samples[:, idx2], c = values, cmap = "viridis", s = 20)
#             plt.xlabel(name1)
#             plt.ylabel(name2)
#             plt.colorbar(sc, label = name)
#             plt.title(f"{name}")
#             plt.savefig(f"./figures/{name}_{name1}_{name2}.png")
#             plt.close()

#         # Now the differences
#         diffs = abs((log_likelihood - log_likelihood_original) / log_likelihood_original)
#         fig, ax = plt.subplots(1, 1, figsize = (8, 8))
#         # TODO log scale?
#         sc = ax.scatter(samples[:, idx1], samples[:, idx2], c = diffs, cmap = "viridis", s = 20)
#         plt.xlabel(name1)
#         plt.ylabel(name2)
#         plt.colorbar(sc, label = "Absolute relative differences log likelihood")
#         plt.title(f"{name}")
#         plt.savefig(f"./figures/log_likelihood_differences_{name1}_{name2}.png")
#         plt.close()
        
        
# def evaluate_original_on_samples(N_samples: int = 50,
#                                  reshape: bool  = True
#                                  ):
#     """Evaluate the original likelihood of a run with the production samples that were generated with during that run.

#     Args:
#         idx1_list (list): _description_
#         idx2_list (list): _description_
#         N_samples (int): default 50: Number of values to sample from the chains, to then compare the original likelihood with the evaluated likelihood.
#     """
    
#     # Load results_production.npz
#     filename = "results_production.npz"
    
#     data = np.load(filename)
#     chains = data["chains"]
#     # chains = np.reshape(chains, (-1, chains.shape[-1]))
#     log_prob = data["log_prob"]
#     log_prob = data["log_prob"]
    
#     print(np.shape(chains))
#     print(np.shape(log_prob))
    
#     # Reshape:
#     if reshape:
#         chains = np.reshape(chains, (-1, chains.shape[-1]))
#         log_prob = np.reshape(log_prob, (-1,))
        
#     print(np.shape(chains))
#     print(np.shape(log_prob))
    
#     # Sample indices to then get the chains and log prob for
#     indices = np.random.choice(np.arange(len(log_prob)), size = N_samples, replace = False)
#     sampled_chains = chains[indices]
#     sampled_log_prob = log_prob[indices]
    
#     # Now, use the likelihood to evaluate the sampled chains
#     with open('likelihood.pickle', 'rb') as f:
#         likelihood = pickle.load(f)
        
#     # Evaluate the true likelihood
#     print("Evaluating the likelihood")
#     my_log_likelihoods = utils.my_evaluate_vmap(likelihood, utils.complete_prior, sampled_chains, original=False)
#     print("Evaluating the original likelihood")
#     my_log_likelihoods_original = utils.my_evaluate_vmap(likelihood, utils.complete_prior, sampled_chains, original=True)
    
#     # Now save
    
#     np.savez("evaluated_likelihoods.npz", sampled_chains = sampled_chains, sampled_log_prob = sampled_log_prob, my_log_likelihoods = my_log_likelihoods, my_log_likelihoods_original = my_log_likelihoods_original)
    
#     return
    
    
# def plot_log_prob_from_file(filename = "evaluated_likelihoods.npz"):
    
#     data = np.load(filename)
#     sampled_chains = data["sampled_chains"]
#     sampled_log_prob = data["sampled_log_prob"]
#     my_log_likelihoods = data["my_log_likelihoods"]
#     my_log_likelihoods_original = data["my_log_likelihoods_original"]
    
#     print("my_log_likelihoods_original")
#     print(my_log_likelihoods_original)
    
#     print("my_log_likelihoods")
#     print(my_log_likelihoods)
    
#     print("sampled_log_prob")
#     print(sampled_log_prob)
    
#     # Get the differences between log likelihoods and print mean
#     diffs_sampled_vs_original = abs((my_log_likelihoods - my_log_likelihoods_original))
#     print(np.mean(diffs_sampled_vs_original))
#     # diffs_sampled_vs_log_prob = abs((my_log_likelihoods - sampled_log_prob) / sampled_log_prob)
#     # print(np.mean(diffs_sampled_vs_log_prob))
    
    
# def main():
#     idx1_list = [2, 3]
#     idx2_list = [4, 5]
#     # plot_sampled_likelihoods(idx1_list, idx2_list)
    
#     # evaluate_original_on_samples()
    
#     plot_log_prob_from_file()
    
# if __name__ == "__main__":
#     main()
#     print("DONE")