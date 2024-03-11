import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
import json
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

# outdirs = ["../../tidal/outdir_NRTv2_test",
#            "../../tidal/outdir_NRTv2_test_Gaussian",
#            "../../tidal/outdir_NRTv2_binsize_500",
#            "../../tidal/outdir_NRTv2_binsize_1000"]

outdirs = ["../../tidal/outdir_NRTv2_06_03"]

naming = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']


def load_runs(outdirs_list: list) -> dict:
    """
    Load the results of the runs from the outdirs_list.
    
    Successful runs has a criterion: see below to see how we judge this. 

    Args:
        outdirs_list (list): List of different outdirs containing injections of NRTv2 (see above for current list)

    Returns:
        dict: Dictionary containing the results of the runs
    """
    results_dict = {}
    for outdir in outdirs_list:
        for subdir in os.listdir(outdir):
            # Check if results_production.npz is there
            if os.path.exists(f"{outdir}/{subdir}/results_production.npz"):
                this_run = f"{outdir}/{subdir}"
                # Make new entry in the dictionary
                results_dict[this_run] = {}
                # Load the results production
                results = np.load(f"{outdir}/{subdir}/results_production.npz")
                # Get global accs, local accs, and log prob and save in the dict
                for key in ["global_accs", "local_accs", "log_prob"]:
                    results_dict[this_run][key] = np.mean(results[key])
                # Also get the injected params. For that, load config.json
                with open(f"{outdir}/{subdir}/config.json") as f:
                    config = json.load(f)
                # Get the true params
                for key in naming:
                    results_dict[this_run][key] = config[key]
                # Indicate whether the run was successful or not
                results_dict[this_run]['success'] = results_dict[this_run]['log_prob'] < 5e3
                # Also load in the network_snr.txt float value
                with open(f"{outdir}/{subdir}/network_snr.txt") as f:
                    results_dict[this_run]['network_snr'] = float(f.read())
                
    return results_dict

def get_successful_runs(results_dict: dict) -> list:
    """
    Criterion for successful run: the global acceptance is above 0.1

    Args:
        results_dict (dict): Dictionary containing the results of the runs

    Returns:
        list: List of the keys with successful runs
    """
    
    successful_runs_keys = []
    for key in results_dict:
        if results_dict[key]['success']:
            successful_runs_keys.append(key)
    total_counter = len(list(results_dict.keys()))
    nb_successful_runs = len(successful_runs_keys)
    print(f"Number of successful runs: {nb_successful_runs} / {total_counter} ({np.round(100 * (nb_successful_runs / total_counter), 4)})")
    return successful_runs_keys

def make_scatterplot(results_dict: dict, 
                     param_name_1: int, 
                     param_name_2: int,
                     ) -> None:
    
    plt.figure(figsize=(10, 10))
    for key in results_dict:
        if results_dict[key]['success']:
            plt.scatter(results_dict[key][param_name_1], results_dict[key][param_name_2], color='green')
        else:
            plt.scatter(results_dict[key][param_name_1], results_dict[key][param_name_2], color='red')
    plt.xlabel(param_name_1)
    plt.ylabel(param_name_2)
    if param_name_1 == "log_prob":
        plt.xscale('log')
    if param_name_2 == "log_prob":
        plt.yscale('log')
    plt.savefig(f"./figures/scatterplot_{param_name_1}_{param_name_2}.png")
    plt.close()
    
    return

def main():
    
    ### Analyze each one separately
    
    for outdir in outdirs:
        print(outdir)
        results_dict = load_runs([outdir]) # note: has to be a list for now
        get_successful_runs(results_dict)
        
    # ## Make some scatterplots
    # results_dict = load_runs(outdirs)
    # successful_runs_keys = get_successful_runs(results_dict)
    
    # print(successful_runs_keys)
    
    # make_scatterplot(results_dict, "s1_z", "network_snr")
    # make_scatterplot(results_dict, "log_prob", "network_snr")
    # make_scatterplot(results_dict, "log_prob", "lambda_1")
    # make_scatterplot(results_dict, "log_prob", "lambda_2")
    # make_scatterplot(results_dict, "log_prob", "lambda_2")
    # make_scatterplot(results_dict, "M_c", "d_L")
    
    # ### Check the log prob of two runs
    # runs = ["../../tidal/test_outdir/injection_1_NRTv2_original",
    #         "../../tidal/test_outdir/injection_1_NRTv2_binsize_1000"]
    
    # print(f"Comparing runtime of two runs: {runs[0]} and {runs[1]}")
    
    # for run in runs:
    #     print(run)
    #     with open(run + "/runtime.txt") as f:
    #         print(f.read())
            
    # print(f"Comparing log prob of two runs: \n {runs[0]}, and \n {runs[1]}")
    
    # for run in runs:
    #     print(run)
    #     filename = run + "/results_production.npz"
    #     results = np.load(filename)
    #     log_prob = np.mean(results['log_prob'])
    #     print(log_prob)
            
    
if __name__ == "__main__":
    main()