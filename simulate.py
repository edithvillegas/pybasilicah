#from os import X_OK
import pandas as pd
import numpy as np
import torch
import pyro.distributions as dist
import aux

# load data
my_path = "/home/azad/Documents/thesis/SigPhylo/cosmic/"
beta_file = "cosmic_catalogue.csv"
beta_full = pd.read_csv(my_path + beta_file)
counts, signature_names, contexts = aux.get_signature_profile(beta_full)


################################################################################
######################### creating dummy inputs ################################
################################################################################

# creating dummy relative exposure matrix

#alpha = pd.read_csv("data/dummy_alpha.csv", header=None)
alpha = torch.tensor([
    [0.35, 0.50, 0.15],
    [0.52, 0.43, 0.05],
    [0.51, 0.45, 0.04],
    [0.02, 0.03, 0.95],
    [0.23, 0.46, 0.31]
    ])

# selecting the fixed and denovo signatures
beta_fixed = counts[[0, 2]]
beta_denovo = counts[[7]]
beta = torch.cat((beta_fixed, beta_denovo), axis=0)
k_denovo = beta_denovo.size()[0]

# creating theta vector as total number of mutations in branches
theta = [1200, 3600, 2300, 1000, 1900]

################################################################################
################################################################################
################################################################################


# simulate data
def simulate():

    # number of branches
    num_samples = alpha.size()[0]
    #num_samples = alpha.shape[0]

    # create initial mutational catalogue
    M = torch.zeros([num_samples, 96])

    for i in range(num_samples):

        # selecting branch i
        p = alpha[i]
        #p = torch.tensor(alpha[i])

        # iterate for number of the mutations in branch i
        for k in range(theta[i]):

            # sample signature profile index from categorical data
            a = dist.Categorical(p).sample().item()
            b = beta[a]

            # sample mutation feature index for corresponding signature profile from categorical data
            j = dist.Categorical(b).sample().item()

            # add +1 to the mutation feature in position j in branch i
            M[i, j] += 1


    #################################################################
    # export generated mutations catalogue to CSV file
    m_np = np.array(M)
    m_df = pd.DataFrame(m_np)
    int_df = m_df.astype(int)
    int_df.to_csv('data/sim_catalogue.csv', index=False, header=False)
    #################################################################
    
    return M, beta_fixed, k_denovo
    # M : tensor
    # beta_fixed : tensor
    # k_denovo : int

