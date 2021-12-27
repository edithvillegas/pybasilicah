import aux_func
import pandas as pd
import torch
import inference
import model

my_path = "/home/azad/Documents/thesis/SigPhylo/data/"
data_file = "data_sigphylo.csv"
aging_file = "beta_aging.csv"

# load data
M = pd.read_csv(my_path + data_file)
beta_aging = pd.read_csv(my_path + aging_file)

# get counts and contexts
M_counts = aux_func.get_phylogeny_counts(M)
beta_counts, signature_names, contexts = aux_func.get_signature_profile(beta_aging)

# define adjacency matrix
A = torch.tensor([[1,1,0,0,0],[1,1,1,1,0],[0,1,1,1,0],[0,1,1,1,1],[0,0,0,1,1]])

params = {"k_denovo" : 1, "beta_fixed" : beta_counts, "A" : A, "lambda": 0.5}

params, alphas, betas = inference.full_inference(M_counts,params, lr = 0.05, steps_per_iteration = 500, num_iterations = 20)

alpha, beta = aux_func.get_alpha_beta(params)

print("alphas :", alphas)
#print("betas :", betas)
