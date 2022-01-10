import aux
import pandas as pd
import torch
import infer
import simulate
import visualize


################################################################
###################### start loading data ######################
################################################################
# mutations cataloge
# beta fixed signature profiles
# no. of inferable signatures profiles

###################### existing data ######################
my_path = "/home/azad/Documents/thesis/SigPhylo/data/"
data_file = "data_sigphylo.csv"
aging_file = "beta_aging.csv"
M = pd.read_csv(my_path + data_file)    # pandas dataframe
beta_aging = pd.read_csv(my_path + aging_file)
M_counts = aux.get_phylogeny_counts(M) # torch tensor
beta_counts, signature_names, contexts = aux.get_signature_profile(beta_aging)
k_denovo = 1

###################### simulated data #####################
M_counts, beta_counts, k_denovo = simulate.simulate()

###################### define adjacency matrix ############
A = torch.tensor([[1,1,0,0,0],[1,1,1,1,0],[0,1,1,1,0],[0,1,1,1,1],[0,0,0,1,1]])

################################################################
###################### end loading data ########################
################################################################

params = {"k_denovo" : k_denovo, "beta_fixed" : beta_counts, "A" : A, "lambda": 0.5}

params, alphas, betas = infer.full_inference(M_counts, params, lr = 0.05, steps_per_iteration = 500, num_iterations = 20)

alpha, beta = aux.get_alpha_beta(params)

print("\nalpha : \n", alpha, "\n")
print("beta : \n", beta, "\n")

# visualize the alpha values among iterations
# 1st arg : list of alphas
# 2nd arg : branch number (starts from 1)
visualize.alpha(alphas, 1)




