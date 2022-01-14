import numpy as np
import pandas as pd
import torch
import infer
import utilities


# ====== LOAD DATA & PARAMETERS =====================================
'''
M               :   mutations cataloge
beta_fixed      :   fixed signature profiles
A               :   adjacency matrix
k_denovo        :   no. of inferable signatures profiles
hyper_lambda    :   lambda parameter
lr              :   lr
steps_per_iter  :   steps per iteration
max_num_iter    :   maximum number of iterations
'''

input = {
    "M_path" : "/home/azad/Documents/thesis/SigPhylo/data/data_sigphylo.csv",
    "beta_fixed_path" : "/home/azad/Documents/thesis/SigPhylo/data/beta_aging.csv",
    "A_path" : "/home/azad/Documents/thesis/SigPhylo/data/A.csv",
    "k_denovo" : 1,
    "hyper_lambda" : 0.6,
    "lr" : 0.05,
    "steps_per_iter" : 500,
    "max_iter" : 100
}

params, alphas, betas = infer.full_inference(input)