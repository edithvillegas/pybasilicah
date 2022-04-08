import torch
import utilities
import run
import pandas as pd
import numpy as np


# ============================== [INPUT DATA] =====================================[PASSED]

right = 0
wrong = 0

for i in range(100):
    path = "/home/azad/Documents/thesis/SigPhylo/cosmic/cosmic_catalogue.csv"
    cosmic_df = pd.read_csv(path, index_col=0)
    input_data = utilities.input_generator(num_samples=5, num_sig=4, cosmic_path=path)
    M = input_data["M"]                         # dataframe
    A = input_data["alpha"]                 # dataframe
    B = input_data["beta"]                   # dataframe
    B_test = input_data["beta_fixed_test"]   # dataframe
    overlap = input_data["overlap"]             # int
    extra = input_data["extra"]                 # int

    params = {
        "M" :               torch.tensor(M.values).float(), 
        "beta_fixed" :      torch.tensor(B_test.values).float(), 
        "lr" :              0.05, 
        "steps_per_iter" :  500
        }

    theta = np.array(torch.sum(params["M"], axis=1))

    k_list = [1, 2, 3, 4, 5]

    # ============================== [RUN] ============================================

    k_inf, A_inf, B_inf = run.multi_k_run(params, k_list)
    B_test_sub = utilities.fixedFilter(A_inf, B_test, theta)    # list
    B_test_new = utilities.denovoFilter(B_inf, path)            # list

    '''
    print("Target Beta          :", list(B.index))
    print("Overlap Signatures   :", overlap)
    print("Extra Signatures     :", extra)
    print("Test Fixed Beta      :", list(B_test.index))
    print("-----------------------------------------------------------------")
    print("Best k (BIC)         :", k_inf)
    print("Filtered Fixed Beta  :", B_test_sub)
    print("new Fixed Signatures :", B_test_new)
    print("-----------------------------------------------------------------")
    '''
    
    B_list = list(B.index)
    B_test_list = B_test_sub + B_test_new
    B_list.sort()
    B_test_list.sort()
    if B_list==B_test_list:
        print("GOOD JOB BRO!")
        right += 1
    else:
        print("FUCK OFF!")
        wrong += 1

print("Right Answer:", right)
print("Wrong Answer:", wrong)

'''
while True:
    print("========================= [Target] ==============================")
    #print(alpha)
    print("=================================================================")
    print("\nRunning ...\n")
    print("========================= [Output] ==============================")
    #print(pd.DataFrame(np.array(alpha_inf)))
    #print("-----------------------------------------------------------------")

    match = utilities.denovoFilter(beta_inf, path)
    if len(match) > 0:
        print("New Fixed Beta(s) :", new_beta_test_list)

    print("=================================================================")
    if stopRun(new_beta_test_list, list(beta_test.index), match):
        break
    
    beta_test = cosmic_df.loc[new_beta_test_list + match]   # dataframe
    params["beta_fixed"] = torch.tensor(beta_test.values).float()

#--------------------------------------------------------------------------------
# Run Model
#--------------------------------------------------------------------------------

if exceed == len(b):
    print("good job! Sir")
else:
    print("such a bloody shit!")
'''

