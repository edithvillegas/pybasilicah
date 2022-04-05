import torch
import utilities
import run
import pandas as pd
import numpy as np

# ===============================================================================
# ============================== INPUT DATA =====================================
# ===============================================================================
num_samples = 5
k_fixed = 3
k_denovo = 1
cosmic_path = "/home/azad/Documents/thesis/SigPhylo/cosmic/cosmic_catalogue.csv"
#--------------------------------------------------------------------------------
data = utilities.input_generator(num_samples, k_fixed, k_denovo, cosmic_path)
#--------------------------------------------------------------------------------
M = data["M"]                               # dtype: dataframe
beta_fixed_test = data["beta_fixed_test"]   # dtype: dataframe
#--------------------------------------------------------------------------------
alpha_target = data["alpha"]                # dtype: dataframe
beta_fixed_target = data["beta_fixed"]      # dtype: dataframe
beta_denovo_target = data["beta_denovo"]    # dtype: dataframe
overlap = data["overlap"]                   # dtype: int
exceed = data["exceed"]                     # dtype: int
#--------------------------------------------------------------------------------
k_list = [1, 2, 3, 4, 5]                    # dtype: list
#--------------------------------------------------------------------------------
params = {
    "M"                 : torch.tensor(M.values).float(),               # dtype: tensor
    "beta_fixed"        : torch.tensor(beta_fixed_test.values).float(), # dtype: tensor
    "lr"                : 0.05, 
    "steps_per_iter"    : 500
    }

# ===============================================================================
# ===============================================================================
# ===============================================================================

print("======================================================")
print("Target ===============================================")
print("======================================================")
print(alpha_target)
print("------------------------------------------------------")
print("Target Fixed Beta  :", list(beta_fixed_target.index))
print("Target Denovo Beta :", list(beta_denovo_target.index))
print("======================================================")

print("\nRunning ...\n")
k_best, alpha_inferred, beta_inferred = run.multi_k_run(params, k_list)

print("======================================================")
print("Output ===============================================")
print("======================================================")
print(pd.DataFrame(np.array(alpha_inferred)))
print("------------------------------------------------------")
print("No. of Overlapped Signatures :", overlap)
print("No. of Exceeded Signatures   :", exceed)
print("best k (BIC) :", k_best)
print("Test Fixed Beta :", list(beta_fixed_test.index))
print("======================================================")

filtered_fixed_list = utilities.fixedFilter(alpha_inferred, beta_fixed_test)
new_fixed_list = utilities.denovoFilter(beta_inferred, cosmic_path)
print("Selected Fixed Signatures:", filtered_fixed_list)
print("new Fixed Signatures:", new_fixed_list)

print("\nRunning ...\n")
signature_names, mutation_features, beta_fixed_test = utilities.beta_read_name(filtered_fixed_list + new_fixed_list, cosmic_path)
params["beta_fixed"] = torch.tensor(beta_fixed_test.values).float()
k_best, alpha_inferred, beta_inferred = run.multi_k_run(params, k_list)

print("======================================================")
print("Output ===============================================")
print("======================================================")
print(pd.DataFrame(np.array(alpha_inferred)))
print("------------------------------------------------------")
print("No. of Overlapped Signatures :", overlap)
print("No. of Exceeded Signatures   :", exceed)
print("best k (BIC) :", k_best)
print("Test Fixed Beta :", list(beta_fixed_test.index))
print("======================================================")

'''
#--------------------------------------------------------------------------------
# Run Model
#--------------------------------------------------------------------------------

def stopRun(new_fixed_list, beta_fixed_test, new_denovo_list):
    if len(new_fixed_list)==len(list(beta_fixed_test.index)) and len(new_denovo_list)==0:
        return "stop"
    else:
        return "continue"

while True:
    print("filtered", filtered_fixed_list)
    print("new", new_fixed_list)
    if stopRun(new_fixed_list, beta_fixed_test, new_fixed_list)=="stop":
        break

    _, _, params["beta_fixed"] = utilities.beta_read_name(filtered_fixed_list + new_fixed_list, cosmic_path)

if exceed == len(b):
    print("good job! Sir")
else:
    print("such a bloody shit!")
'''

