"""
Trying to make the pp plot with bilby to see how it goes
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import bilby 
from bilby.core.result import Result

OUTDIR = "../tidal/outdir_TaylorF2_part3/"
OUTDIR = os.path.abspath(OUTDIR)
NAMING = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']

def create_bilby_result(outdir,
                        convert_cos_sin: bool = True,
                        thinning = 100):
    """
    Create a Bilby result object from a given file with a jim run

    """
    
    # Load the true injection parameters from the config.json file
    injection_parameters = {}
    with open(os.path.join(outdir, "config.json"), "r") as file:
        config = json.load(file)
    for name in NAMING:
        injection_parameters[name] = config[name]
        
    # Convert cos and sin if desired
    if convert_cos_sin:
        injection_parameters["iota"] = np.arccos(injection_parameters["cos_iota"])
        del injection_parameters["cos_iota"]
        injection_parameters["dec"] = np.arccos(injection_parameters["sin_dec"])
        del injection_parameters["sin_dec"]
        
    result = Result(outdir=outdir, 
                    label=outdir, 
                    injection_parameters=injection_parameters, 
                    convert_cos_sin=convert_cos_sin)
    
    # Load the posterior samples
    filename_npz = os.path.join(outdir, "results_production.npz")
    # convert_npz_to_json(filename_npz)
    
    data = np.load(filename_npz)
    chains = data["chains"]
    chains = np.reshape(chains, (-1, chains.shape[-1]))
    chains = chains[::thinning]
    
    if convert_cos_sin:
        iota_index = NAMING.index("cos_iota")
        chains[iota_index] = np.arccos(chains[iota_index])
        dec_index = NAMING.index("sin_dec")
        chains[dec_index] = np.arccos(chains[dec_index])
    
    # filename_json = filename_npz.replace(".npz", ".json")
    # result.from_json(filename_json) ### calls init
    
    print(result)
    
    return result

# def convert_npz_to_json(filename_npz, 
#                         thinning: int = 100, 
#                         convert_cos_sin: bool = True):
#     data = np.load(filename_npz)
    
#     chains = data["chains"]
#     chains = np.reshape(chains, (-1, chains.shape[-1]))
    
#     if convert_cos_sin:
#         iota_index = NAMING.index("cos_iota")
#         chains[iota_index] = np.arccos(chains[iota_index])
#         dec_index = NAMING.index("sin_dec")
#         chains[dec_index] = np.arccos(chains[dec_index])
    
#     data_dict = {}
#     for (name, param_values) in zip(NAMING, chains.T):
#         data_dict[name] = list(param_values[::thinning])
#     filename_json = filename_npz.replace(".npz", ".json")
    
    
#     print("data_dict")
#     for key in data_dict.keys():
#         print(key, len(data_dict[key]))
    
#     with open(filename_json, "w") as file:
#         json.dump(data_dict, file)
        
#     print("Saved json file")
    
        
#     return
    
    
def main():
    # Load the result
    outdir = os.path.join(OUTDIR, "injection_1")
    result = create_bilby_result(outdir)
    
    plt.show()
    
if __name__ == "__main__":
    main()