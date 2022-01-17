import numpy as np
import pandas as pd
import torch
import infer
import utilities

# Questions?
# 1. non-negativity and normalization are done also inside the variational inference calculation
# 2. transfer coeff multiplied by pure alpha or preprocessed (non-negativity and normalizing)


input = {
    "M_path" : "/home/azad/Documents/thesis/SigPhylo/data/data_sigphylo.csv",
    "beta_fixed_path" : "/home/azad/Documents/thesis/SigPhylo/data/beta_aging.csv",
    "A_path" : "/home/azad/Documents/thesis/SigPhylo/data/A.csv",
    "k_denovo" : 1,

    "hyper_lambda" : 0.6,
    "lr" : 0.05,
    "steps_per_iter" : 500,
    "max_iter" : 100,
    "epsilon" : 0.05
    }

#infer.full_inference(input)


# different lambdas
hyper_lambda = [0, 0.3, 0.5, 0.8, 1]

def run_over_lambda(hyper_lambda):
    for i in hyper_lambda:
        print("lambda =", i)
        input = {
            "M_path" : "/home/azad/Documents/thesis/SigPhylo/data/data_sigphylo.csv",
            "beta_fixed_path" : "/home/azad/Documents/thesis/SigPhylo/data/beta_aging.csv",
            "A_path" : "/home/azad/Documents/thesis/SigPhylo/data/A.csv",
            "k_denovo" : 1,

            "hyper_lambda" : i,
            "lr" : 0.05,
            "steps_per_iter" : 500,
            "max_iter" : 100,
            "epsilon" : 0.05
            }
        infer.full_inference(input)

run_over_lambda(hyper_lambda)