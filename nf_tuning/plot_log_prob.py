# import utils 
# import os

# for subdir in os.listdir("./outdir/"):
#     if "7" in subdir:
#         print(subdir)
#         # Check if they have a results_training.npz file
#         if not os.path.isfile(f"./outdir/{subdir}/results_training.npz"):
#             utils.plot_log_prob_from_file(f"./outdir/{subdir}/", which_list=["production"])
#         else:
#             utils.plot_log_prob_from_file(f"./outdir/{subdir}/")