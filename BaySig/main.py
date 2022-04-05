import batch
import torch
import utilities
import numpy as np
import random
import pandas as pd
import pyro.distributions as dist
import pyro
import torch.nn.functional as F
import svi
import run

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


#--------------------------------------------------------------------------------
# Run Model
#--------------------------------------------------------------------------------
k_best, alpha_inferred, beta_inferred = run.multi_k_run(params, k_list)

filtered_fixed_list = utilities.fixedFilter(alpha_inferred, beta_fixed_test)
new_fixed_list = utilities.denovoFilter(beta_inferred, cosmic_path)

print("filtered", filtered_fixed_list)
print("new", new_fixed_list)

def stopRun(new_fixed_list, beta_fixed_test, new_denovo_list):
    if len(new_fixed_list)==len(beta_fixed_test) and len(new_denovo_list)==0:
        return "stop"
    else:
        return "continue"



'''
def BaySiCo(M, beta_fixed_test, k_list):
print("----------------------------------")
print("expected alpha :\n", alpha_target)
print("----------------------------------")
print("inferred alpha :\n", pd.DataFrame(np.array(alpha_inferred)))
print("----------------------------------")
print("Target Fixed Beta :", list(beta_fixed_target.index))
print("Target Denovo Beta :", list(beta_denovo_target.index))
print("----------------------------------")
print("No. of Overlapped Signatures :", overlap)
print("No. of Exceeded Signatures :", exceed)
print("best k (BIC) :", k_best)
print("Test Fixed Beta :", list(beta_fixed_test.index), "\n")
print("----------------------------------")
if exceed == len(b):
    print("good job! Sir")
else:
    print("such a bloody shit!")
'''

