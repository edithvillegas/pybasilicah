from turtle import shape
from typing import Counter
import torch
import utilities
import run
import pandas as pd
import numpy as np
import os
import shutil
import json



def BaySiCo(M, B_fixed, k_list, cosmic_path):
    # M ------------ dataframe
    # B_fixed ------ dataframe
    # k_list ------- list
    # cosmic_path -- str
    cosmic_df = pd.read_csv(cosmic_path, index_col=0)
    theta = np.sum(M.values, axis=1)
    params = {
        "M" :               torch.tensor(M.values).float(), 
        "beta_fixed" :      torch.tensor(B_fixed.values).float(), 
        "lr" :              0.05, 
        "steps_per_iter" :  500
        }

    counter = 1
    while True:
        print("Loop", counter)
        
        k_inf, A_inf, B_inf = run.multi_k_run(params, k_list)

        #print("k_denovo:", k_inf)

        B_fixed_sub = utilities.fixedFilter(A_inf, B_fixed, theta)    # list
        
        if k_inf > 0:
            B_fixed_new = utilities.denovoFilter(B_inf, cosmic_path) # list
        else:
            B_fixed_new = []

        if utilities.stopRun(B_fixed_sub, list(B_fixed.index), B_fixed_new):
            signatures_inf = []
            for k in range(k_inf):
                signatures_inf.append("Unknown"+str(k+1))
            signatures = list(B_fixed.index) + signatures_inf
            mutation_features = list(B_fixed.columns)

            A_np = np.array(A_inf)
            A_df = pd.DataFrame(A_np, columns=signatures)

            if B_inf=="NA":
                B_full = params["beta_fixed"]
            else:
                B_full = torch.cat((params["beta_fixed"], B_inf), axis=0)
            
            B_np = np.array(B_full)
            B_df = pd.DataFrame(B_np, index=signatures, columns=mutation_features)

            return A_df, B_df

        B_fixed = cosmic_df.loc[B_fixed_sub + B_fixed_new]
        params["beta_fixed"] = torch.tensor(B_fixed.values).float()
        counter += 1



# ============================== [INPUT DATA] =====================================[PASSED]

def batchRun(num_iter, cos_path, save_path):

    output = {}
    for i in range(num_iter):
        input_data = utilities.input_generator(num_samples=5, num_sig=4, cosmic_path=cos_path)
        M = input_data["M"]                         # dataframe
        A = input_data["alpha"]                     # dataframe
        B = input_data["beta"]                      # dataframe
        B_fixed = input_data["beta_fixed_test"]     # dataframe
        overlap = input_data["overlap"]             # int
        extra = input_data["extra"]                 # int

        k_list = [0, 1, 2, 3, 4, 5]

        A_inf, B_inf = BaySiCo(M, B_fixed, k_list, cosmic_path=cos_path)

        #print("---------------------- INPUT -------------------------------------------")
        targetSig = list(B.index)
        inputSig = list(B_fixed.index)
        inferredSig = list(B_inf.index)
        #print("Target List      :", targetSig)
        #print("Input List       :", inputSig)
        #print("Inferred List    :", inferredSig)
        TP = len(list(set(targetSig).intersection(inferredSig)))
        FP = len(list(set(inferredSig) - set(targetSig)))
        TN = len(list((set(inputSig) - set(targetSig)) - set(inferredSig)))
        FN = len(list(set(targetSig) - set(inferredSig)))
        Accuracy = (TP + TN)/(TP + TN + FP + FN)
        Precision = (TP)/(TP + FP)
        Recall = (TP)/(TP + FN)
        F1 =  2 / (1/Precision + 1/Recall)

        #print("Accuracy     :", Accuracy)
        #print("Precision    :", Precision)
        #print("Recall       :", Recall)
        #print("F1           :", F1)

        #print("A target:\n", A)
        #print("A inferred:\n", A_inf)
        if A.shape == A_inf.shape:
            mse = (np.square(A.values - A_inf.values)).mean()
            #print(mse)

        else:
            #print("NO!")
            pass

        output[str(i+1)] = {
            "alpha_target"      : A.values,         # ndarrays
            "beta_target"       : B.values,         # ndarrays
            "beta_fixed"        : B_fixed.values,   # ndarrays
            "alpha_inferred"    : A_inf.values,     # ndarrays
            "beta_inferred"     : B_inf.values,     # ndarrays
            "Accuracy"          : float(Accuracy),         # float
            "Precision"         : float(Precision),        # float
            "Recall"            : float(Recall),           # float
            "F1"                : float(F1),               # float
            "alpha_mse"         : float(mse)               # float
            }

    #------------------------------------------------------------------------------------
    # create new directory (overwrite if exist) and export data as JSON file
    #------------------------------------------------------------------------------------
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    #print("New directory made!")

    with open(save_path + "/output.json", 'w') as outfile:
        json.dump(output, outfile, cls=utilities.NumpyArrayEncoder)
        #print("Exported as JSON file!")
        
    return output


cos_path = "/home/azad/Documents/thesis/SigPhylo/cosmic/cosmic_catalogue.csv"
res = batchRun(2, cos_path, "/home/azad/Documents/thesis/SigPhylo/data/results/baysico")

'''
input_data = utilities.input_generator(num_samples=5, num_sig=4, cosmic_path=cos_path)
M = input_data["M"]                         # dataframe
A = input_data["alpha"]                     # dataframe
B = input_data["beta"]                      # dataframe
B_fixed = input_data["beta_fixed_test"]     # dataframe
overlap = input_data["overlap"]             # int
extra = input_data["extra"]                 # int
k_list = [0, 1, 2, 3, 4, 5]

A_df, B_df = BaySiCo(M, B_fixed, k_list, cos_path)
'''
