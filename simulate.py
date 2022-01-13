import pandas as pd
import numpy as np
import torch
import pyro.distributions as dist

# load full signature profiles
path = "/home/azad/Documents/thesis/SigPhylo/cosmic/cosmic_catalogue.csv"
cosmic_catalogue = pd.read_csv(path, index_col=0)



# simulate data
def catalogue(fixed_signatures, denovo_signatures):

    ################# load dummy alpha #############################################
    alpha_path = "data/simulated/dummy_alpha.csv"
    df = pd.read_csv(alpha_path, header=None)   # dtype:Pandas.DataFrame
    alpha = torch.tensor(df.values)             # dtype:torch.Tensor

    ################# load dummy theta #############################################
    theta_path = "data/simulated/dummy_theta.csv"
    df = pd.read_csv(theta_path, header=None)   # dtype:Pandas.DataFrame
    theta = df.values.tolist()[0]               # dtype:list
    #theta = torch.tensor(df.values)            # dtype:torch.Tensor

    ################# load full beta ###############################################
    beta_path = "/home/azad/Documents/thesis/SigPhylo/cosmic/cosmic_catalogue.csv"
    beta_full = pd.read_csv(beta_path, index_col=0)

    # get fixed signature profiles
    beta_fixed = beta_full.loc[fixed_signatures] # Pandas.DataFrame
    beta_fixed = beta_fixed.values          # numpy.ndarray
    beta_fixed = torch.tensor(beta_fixed)   # torch.Tensor

    # get denovo signature profiles
    beta_denovo = beta_full.loc[denovo_signatures] # Pandas.DataFrame
    beta_denovo = beta_denovo.values          # numpy.ndarray
    beta_denovo = torch.tensor(beta_denovo)   # torch.Tensor

    beta = torch.cat((beta_fixed, beta_denovo), axis=0)
    
    # number of branches
    num_samples = alpha.size()[0]

    # initialize mutational catalogue with zeros
    M = torch.zeros([num_samples, 96])

    for i in range(num_samples):

        # selecting branch i
        p = alpha[i]

        # iterate for number of the mutations in branch i
        for k in range(theta[i]):

            # sample signature profile index from categorical data
            a = dist.Categorical(p).sample().item()
            b = beta[a]

            # sample mutation feature index for corresponding signature profile from categorical data
            j = dist.Categorical(b).sample().item()

            # add +1 to the mutation feature in position j in branch i
            M[i, j] += 1

    ####### export results to CSV files #############################
    m_np = np.array(M)
    m_df = pd.DataFrame(m_np)
    int_df = m_df.astype(int)
    int_df.to_csv('data/simulated/sim_catalogue.csv', index=False, header=False)

    fix_np = np.array(beta_fixed)
    fix_df = pd.DataFrame(fix_np)
    #fix_int_df = fix_df.astype(int)
    fix_df.to_csv('data/simulated/sim_beta_fixed.csv', index=False, header=False)

    denovo_np = np.array(beta_denovo)
    denovo_df = pd.DataFrame(denovo_np)
    #denovo_int_df = denovo_df.astype(int)
    denovo_df.to_csv('data/simulated/sim_beta_denovo.csv', index=False, header=False)
    #################################################################
    
    return M, beta_fixed, beta_denovo
    # M : dtype: torch.Tensor
    # beta_fixed  : dtype: torch.Tensor
    # beta_denovo : dtype: torch.Tensor



'''
===================== STORAGE =====================
# (dtype:torch.tensor)
alpha = torch.tensor([
    [0.35, 0.50, 0.15],
    [0.52, 0.43, 0.05],
    [0.51, 0.45, 0.04],
    [0.02, 0.03, 0.95],
    [0.23, 0.46, 0.31]
    ])
theta = [1200, 3600, 2300, 1000, 1900]
'''