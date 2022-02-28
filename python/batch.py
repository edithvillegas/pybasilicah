
import json
import numpy as np
import os
import utilities
import shutil
import SigPhylo
import torch

arg_list = {
    "k_list"        : [1, 2], 
    "lambda_list"   : [0, 0.1, 0.2],
    "dir"           : "/home/azad/Documents/thesis/SigPhylo/data/results/realFull",
    "sim"           : False
    }

#====================================================================================
# BATCH RUN -------------------------------------------------------------------------
#====================================================================================

def batch_run(arg_list):

    sim = arg_list["sim"]

    # OK
    input = {
        "k_list"            : arg_list["k_list"], 
        "lambda_list"       : arg_list["lambda_list"], 
        "lr"                : 0.05, 
        "steps_per_iter"    : 500, 
        "max_iter"          : 50, 
        "epsilon"           : 0.001, 
        "dir"               : arg_list["dir"]
    }

    if sim:
        #============== simulation ==================================================
        # OK
        M, alpha, beta_fixed, beta_denovo, A = utilities.generate_data()

        input["M"]                 = torch.tensor(M.values).float() # dataframe to tensor
        input["beta_fixed"]        = torch.tensor(beta_fixed.values).float() # dataframe to tensor
        input["A"]                 = torch.tensor(A.values) # dataframe to tensor
        
        #-------------------------- To JSON ----------------------------------------
        # beta fixed info (OK)
        input["beta_fixed_names"]  = list(beta_fixed.index)
        input["mutation_features"] = list(beta_fixed.columns)

        # alpha expected & info (OK)
        input["alpha_expected"]        = alpha.values # dataframe to numpy
        input["alpha_expected_names"]  = list(alpha.columns)
        
        # beta expected & info (OK)
        input["beta_expected"]         = beta_denovo.values # dataframe to numpy
        input["beta_expected_names"]   = list(beta_denovo.index)
        #input["mutation_features"]     = list(beta_denovo.columns)
        #============================================================================

    else:
        #============== real ========================================================
        # OK
        M_path = "/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv"
        #beta_fixed_path = "/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv"
        A_path = "/home/azad/Documents/thesis/SigPhylo/data/real/A.csv"
        expected_beta_path = "/home/azad/Documents/thesis/SigPhylo/data/real/expected_beta_2.csv"

        input["M"]                  = utilities.M_read_csv(M_path)[1]       # tensor
        input["beta_fixed"]         = utilities.beta_read_name(["SBS5"])[2] # tensor (modify if needed)
        input["A"]                  = utilities.A_read_csv(A_path)          # tensor

        #-------------------------- only for JSON ----------------------------------------
        # beta fixed info (OK)
        input["beta_fixed_names"]   = utilities.beta_read_name(["SBS5"])[0] # list (modify if needed)
        input["mutation_features"]  = utilities.beta_read_name(["SBS5"])[1] # list (modify if needed)

        # alpha expected & info (OK)
        input["alpha_expected"]        = "NA"
        input["alpha_expected_names"]  = "NA"

        # beta expected & info (OK)
        input["beta_expected"]          = np.array(utilities.beta_read_csv(expected_beta_path)[2]) # SBS1 & missing (numpy)
        input["beta_expected_names"]    = utilities.beta_read_csv(expected_beta_path)[0] # list
        #====================================================================================

    input_data = {
        "M"                     : np.array(input["M"]), 
        "beta_fixed"            : np.array(input["beta_fixed"]), 
        "A"                     : np.array(input["A"]), 

        "beta_fixed_names"      : input["beta_fixed_names"], # list

        "alpha_expected"        : input["alpha_expected"], # numpy
        "alpha_expected_names"  : input["alpha_expected_names"], # list

        "beta_expected"         : input["beta_expected"], # numpy
        "beta_expected_names"   : input["beta_expected_names"], # list

        "mutation_features"     : input["mutation_features"] # list
        }

    #------------------------------------------------------------------------------------
    # run SigPhylo with corresponding input data
    #------------------------------------------------------------------------------------

    # params includes 9 paramenters (will add 2 more later) (OK)
    params = {
        "M"                 : input["M"], 
        "beta_fixed"        : input["beta_fixed"], 
        "A"                 : input["A"], 
        "lr"                : input["lr"], 
        "steps_per_iter"    : input["steps_per_iter"], 
        "max_iter"          : input["max_iter"], 
        "epsilon"           : input["epsilon"]
        }

    output_data = {}
    i = 1
    for k in input["k_list"]:
        for landa in input["lambda_list"]:

            print("k_denovo =", k, "| lambda =", landa)

            params["k_denovo"] = k
            params["lambda"] = landa

            output_data[str(i)] = SigPhylo.single_run(params)
            i += 1

    output = {"input":input_data, "output": output_data}


    #------------------------------------------------------------------------------------
    # create new directory (overwrite if exist) and export data as JSON file
    #------------------------------------------------------------------------------------
    new_dir = input["dir"]
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir)

    with open(new_dir + "/output.json", 'w') as outfile:
        json.dump(output, outfile, cls=utilities.NumpyArrayEncoder)

