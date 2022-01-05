import pandas as pd
import torch
import pyro.distributions as dist
import aux

# load data
my_path = "/home/azad/Documents/thesis/SigPhylo/cosmic/"
beta_file = "cosmic_catalogue.csv"
beta_full = pd.read_csv(my_path + beta_file)
counts, signature_names, contexts = aux.get_signature_profile(beta_full)

# selected signature profiles
beta = counts[[0, 2, 7, 8, 24]]    

# create the alpha matrix
alpha = torch.tensor([
    [0.35, 0.50, 0.15],
    [0.52, 0.43, 0.05],
    [0.51, 0.45, 0.04],
    [0.02, 0.03, 0.95],
    [0.23, 0.46, 0.31]
    ])

num_samples = alpha.size()[0]               # number of branches
theta = [1200, 3600, 2300, 1000, 1900]      # total number of mutations of the branches

# simulate data
def simulate():

    # create initial mutational catalogue
    M = torch.zeros([num_samples, 96])

    for i in range(num_samples):

        p = alpha[i]
        for k in range(theta[i]):

            # add +1 to mutation feature j in branch i

            # sample signature profile index from categorical data
            a = dist.Categorical(p).sample().item()
            b = beta[a]
            #print("signature", x+1, "selected")

            # sample mutation feature index for corresponding signature profile from categorical data
            j = dist.Categorical(b).sample().item()
            #print("mutation feature", j, "selected")

            # add +1 to the mutation feature in position j in branch i
            M[i, j] += 1
    
    return M

