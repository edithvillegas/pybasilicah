import pandas as pd
import torch
import pyro.distributions as dist
import aux_func

# load data
my_path = "/home/azad/Documents/thesis/SigPhylo/cosmic/"
beta_file = "cosmic_catalogue.csv"
beta_full = pd.read_csv(my_path + beta_file)
counts, signature_names, contexts = aux_func.get_signature_profile(beta_full)

# selected signature profiles
beta = counts[[0, 2, 7]]    

# number of branches
num_samples = 3

# number of mutations
num_mutations = [1200, 3600, 2300]

# create initial mutational catalogue
M = torch.zeros([num_samples, 96])

# create the alpha matrix
alpha = torch.tensor([
    [0.35, 0.50, 0.15],
    [0.02, 0.03, 0.95],
    [0.52, 0.43, 0.05]
    ])

# add one mutation feature in branch i
def generativeModel(i):
    p = alpha[0]
    # sample signature profile index from categorical data
    a = dist.Categorical(p).sample().item()
    b = counts[a]
    #print("signature", x+1, "selected")

    # sample mutation feature index for corresponding signature profile from categorical data
    j = dist.Categorical(b).sample().item()
    #print("mutation feature", j, "selected")

    # add +1 to the mutation feature in position j in branch i
    M[i, j] += 1


# simulate data
def simulate(num_samples, num_mutations):
    for i in range(num_samples):
        for k in range(num_mutations[i]):
            generativeModel(i)
    
    return M
