import numpy as np
import pandas as pd
import torch

import infer
import utilities


###################### start loading data ######################
# mutations cataloge
# beta fixed signature profiles
# no. of inferable signatures profiles

###################### existing data ######################
'''
data_path = "/home/azad/Documents/thesis/SigPhylo/data/data_sigphylo.csv"
M = pd.read_csv(data_path)  # Pandas.DataFrame
M_counts = utilities.get_phylogeny_counts(M) # torch tensor

beta_fixed_path = "/home/azad/Documents/thesis/SigPhylo/data/beta_aging.csv"
beta_fixed = pd.read_csv(beta_fixed_path, index_col=0)
beta_counts, signature_names, contexts = utilities.get_signature_profile(beta_fixed)

k_denovo = 1
'''

###################### simulated data #####################
fixed_signatures = ["SBS1", "SBS3"]
denovo_signatures = ["SBS5"]

M_counts, beta_counts, beta_denovo = utilities.generate_data(fixed_signatures, denovo_signatures)

k_denovo = beta_denovo.size()[0]

###################### define adjacency matrix ############
A = torch.tensor([[1,1,0,0,0],[1,1,1,1,0],[0,1,1,1,0],[0,1,1,1,1],[0,0,0,1,1]])


params = {"k_denovo" : k_denovo, "beta_fixed" : beta_counts, "A" : A, "lambda": 0.9}

params, alphas, betas = infer.full_inference(M_counts, params, lr = 0.05, steps_per_iteration = 500, max_num_iterations = 100)

alpha, beta = utilities.get_alpha_beta(params)
# alpha : tensor (num_samples   X   k)
# beta  : tensor (k_denovo      X   96)

#################################################################
# add the new alpha and beta to the list
a_np = np.array(alpha)
a_df = pd.DataFrame(np.array(alpha))
a_df.to_csv('results/alpha.csv', index=False, header=False)

b_np = np.array(beta)
b_df = pd.DataFrame(b_np)
b_df.to_csv('results/beta.csv', index=False, header=False)
#################################################################

#print("\nalpha : \n", alpha, "\n")
#print("beta : \n", beta, "\n")



