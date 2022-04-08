from typing import Counter
import torch
import utilities
import run
import pandas as pd
import numpy as np


# ============================== [INPUT DATA] =====================================[PASSED]

right = 0
wrong = 0

def BaySiCo():
    path = "/home/azad/Documents/thesis/SigPhylo/cosmic/cosmic_catalogue.csv"
    cosmic_df = pd.read_csv(path, index_col=0)

    input_data = utilities.input_generator(num_samples=5, num_sig=4, cosmic_path=path)
    M = input_data["M"]                         # dataframe
    A = input_data["alpha"]                     # dataframe
    B = input_data["beta"]                      # dataframe
    B_test = input_data["beta_fixed_test"]      # dataframe
    overlap = input_data["overlap"]             # int
    extra = input_data["extra"]                 # int

    params = {
        "M" :               torch.tensor(M.values).float(), 
        "beta_fixed" :      torch.tensor(B_test.values).float(), 
        "lr" :              0.05, 
        "steps_per_iter" :  500
        }

    theta = np.array(torch.sum(params["M"], axis=1))

    k_list = [0, 1, 2, 3, 4, 5]

    print("-----------------------------------------------------------------")
    print("Target Beta          :", list(B.index))
    print("Overlap Signatures   :", overlap)
    print("Extra Signatures     :", extra)
    print("Test Fixed Beta      :", list(B_test.index))
    print("-----------------------------------------------------------------")

    counter = 1
    while True:
        print("Loop", counter)
        # ============================== [RUN] ============================================

        k_inf, A_inf, B_inf = run.multi_k_run(params, k_list)
        B_test_sub = utilities.fixedFilter(A_inf, B_test, theta)    # list
        if k_inf > 0:
            B_test_new = utilities.denovoFilter(B_inf, path)        # list
        else:
            B_test_new = []

        print("B_test_sub:", B_test_sub)
        print("k:", k_inf)
        print("B_test_new:", B_test_new)
        print("-----------------------------------------------------------------")

        if utilities.stopRun(B_test_sub, list(B_test.index), B_test_new):
            break

        B_test = cosmic_df.loc[B_test_sub + B_test_new]
        params["beta_fixed"] = torch.tensor(B_test.values).float()
        counter += 1

    print("\n=================== Final Results ===============================")
    print("Best k (BIC)         :", k_inf)
    print("New Test Fixed Beta  :", B_test_sub + B_test_new)
    print("-----------------------------------------------------------------")

    return k_inf, A_inf, B_inf

k_inf, A_inf, B_inf = BaySiCo()

'''
B_list = list(B.index)
B_test_list = B_test_sub + B_test_new
if set(B_test_list).issubset(B_list) and len(B_test_sub)==overlap:
    print("GOOD JOB BRO!")
    right += 1
else:
    print("FUCK OFF!")
    wrong += 1

print("Right Answer:", right)
print("Wrong Answer:", wrong)
'''

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

