import torch
import utilities
import run
import pandas as pd
import numpy as np


# ============================== [INPUT DATA] =====================================
path = "/home/azad/Documents/thesis/SigPhylo/cosmic/cosmic_catalogue.csv"
cosmic_df = pd.read_csv(path, index_col=0)
input_data = utilities.input_generator(num_samples=5, num_sig=4, cosmic_path=path)

M = input_data["M"]                         # dataframe
alpha = input_data["alpha"]                 # dataframe
beta = input_data["beta"]                   # dataframe
beta_test = input_data["beta_fixed_test"]   # dataframe
overlap = input_data["overlap"]             # int
exceed = input_data["exceed"]               # int

theta = np.array(torch.sum(torch.tensor(M.values).float(), axis=1))

# ============================== [RUN] ============================================

params = {
    "M" :               torch.tensor(M.values).float(), 
    "beta_fixed" :      torch.tensor(beta_test.values).float(), 
    "lr" :              0.05, 
    "steps_per_iter" :  500
    }

k_list = [1, 2, 3, 4, 5]
k_inf, alpha_inf, beta_inf = run.multi_k_run(params, k_list)

print("========================= [Target] ==============================")
#print(alpha)
#print("-----------------------------------------------------------------")
print("Target Beta  :", list(beta.index))
print("Overlapped Signatures :", overlap)
print("Exceeded Signatures   :", exceed)
print("Test Fixed Beta :", list(beta_test.index))
print("=================================================================")

print("\nRunning ...\n")
k_inf, alpha_inf, beta_inf = run.multi_k_run(params, k_list)

print("========================= [Output] ==============================")
#print(pd.DataFrame(np.array(alpha_inf)))
#print("-----------------------------------------------------------------")
print("Best k (BIC) :", k_inf)
beta_test_list = utilities.fixedFilter(alpha_inf, beta_test, theta)
print("Filtered Fixed Beta :", beta_test_list)
print("=================================================================")


beta_test = cosmic_df.loc[beta_test_list]
#beta_tensor = torch.tensor(beta_test.values).float()
#signature_names = list(beta_test.index)
#mutation_features = list(beta_test.columns)


'''
filtered_fixed_list = utilities.fixedFilter(alpha_inf, beta_test)
new_fixed_list = utilities.denovoFilter(beta_inf, path)
print("Selected Fixed Signatures:", filtered_fixed_list)
print("new Fixed Signatures:", new_fixed_list)

print("\nRunning ...\n")

cosmic_df = pd.read_csv(cosmic_path, index_col=0)
beta_fixed_test = cosmic_df.loc[filtered_fixed_list + new_fixed_list]
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

