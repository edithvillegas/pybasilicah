import pandas as pd
import basilica
import simulation
import torch
import torch.nn.functional as F
import PyBaSiLiCa
import random

random.seed(256)

M_path = "/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv"
B_input_path = "/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv"
cosmic_path = "/home/azad/Documents/thesis/SigPhylo/data/cosmic/cosmic_catalogue.csv"
exp_beta_path = "/home/azad/Documents/thesis/SigPhylo/data/real/expected_beta.csv"

M = pd.read_csv(M_path)
B_input = pd.read_csv(B_input_path, index_col=0)
cosmic_df = pd.read_csv(cosmic_path, index_col=0)
k_list = [0, 1, 2, 3, 4, 5]
fixedLimit = 0.05
denovoLimit = 0.9
expected_beta = pd.read_csv(exp_beta_path, index_col=0)

A_inf_df, B_inf_fixed_df, B_inf_denovo_df = basilica.BaSiLiCa(M, B_input, k_list, cosmic_df, fixedLimit, denovoLimit)

print("Alpha:\n",A_inf_df)
print("Beta Fixed:\n", B_inf_fixed_df)
print("Beta Denovo:\n", B_inf_denovo_df)
print("Beta Expected:\n", expected_beta)

