import torch
import utilities
import run
import pandas as pd
import numpy as np




def stopRun(new_list, old_list, denovo_list):
    new_list.sort()
    old_list.sort()
    if new_list==old_list and len(denovo_list)==0:
        return True
    else:
        return False



# ============================== [INPUT DATA] =====================================[PASSED]

path = "/home/azad/Documents/thesis/SigPhylo/cosmic/cosmic_catalogue.csv"

cosmic_df = pd.read_csv(path, index_col=0)

input_data = utilities.input_generator(num_samples=5, num_sig=4, cosmic_path=path)
M = input_data["M"]                         # dataframe
alpha = input_data["alpha"]                 # dataframe
beta = input_data["beta"]                   # dataframe
beta_test = input_data["beta_fixed_test"]   # dataframe
overlap = input_data["overlap"]             # int
exceed = input_data["exceed"]               # int

params = {
    "M" :               torch.tensor(M.values).float(), 
    "beta_fixed" :      torch.tensor(beta_test.values).float(), 
    "lr" :              0.05, 
    "steps_per_iter" :  500
    }

theta = np.array(torch.sum(params["M"], axis=1))

k_list = [1, 2, 3, 4, 5]

# ============================== [RUN] ============================================


k_inf, alpha_inf, beta_inf = run.multi_k_run(params, k_list)

new_beta_test_list = utilities.fixedFilter(alpha_inf, beta_test, theta)

match = utilities.denovoFilter(beta_inf, path)

#beta_test = cosmic_df.loc[new_beta_test_list + match]   # dataframe
#params["beta_fixed"] = torch.tensor(beta_test.values).float()

#print("Target Beta  :", list(beta.index))
#print("Test Beta :", list(beta_test.index))



'''
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

params = {
    "M" :               torch.tensor(M.values).float(), 
    "beta_fixed" :      torch.tensor(beta_test.values).float(), 
    "lr" :              0.05, 
    "steps_per_iter" :  500
    }

theta = np.array(torch.sum(params["M"], axis=1))

k_list = [1, 2, 3, 4, 5]


# ============================== [RUN] ============================================

while True:

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
    new_beta_test_list = utilities.fixedFilter(alpha_inf, beta_test, theta)
    print("Filtered Fixed Beta(s) :", new_beta_test_list)

    match = utilities.denovoFilter(beta_inf, path)
    if len(match) > 0:
        print("New Fixed Beta(s) :", new_beta_test_list)



    print("=================================================================")

    if stopRun(new_beta_test_list, list(beta_test.index), match):
        break
    
    beta_test = cosmic_df.loc[new_beta_test_list + match]   # dataframe
    params["beta_fixed"] = torch.tensor(beta_test.values).float()






filtered_fixed_list = utilities.fixedFilter(alpha_inf, beta_test)
new_fixed_list = utilities.denovoFilter(beta_inf, path)
print("Selected Fixed Signatures:", filtered_fixed_list)
print("new Fixed Signatures:", new_fixed_list)

#--------------------------------------------------------------------------------
# Run Model
#--------------------------------------------------------------------------------


if exceed == len(b):
    print("good job! Sir")
else:
    print("such a bloody shit!")
'''

