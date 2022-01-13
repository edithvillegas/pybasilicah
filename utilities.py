import torch
import numpy as np
import pandas as pd

# load data

# input: dataframe - output: tensor
def get_phylogeny_counts(M):
    M = M.values                                # convert to numpy array
    M = torch.tensor(np.array(M, dtype=float))  # convert to tensor
    M = M.float()
    return M   

'''
def get_signature_profile(beta):
    contexts = list(beta.columns[1:])
    signature_names = list(beta.values[:, 0])
    counts = beta.values[:,1:]
    counts = torch.tensor(np.array(counts, dtype=float))
    counts = counts.float()
    return counts, signature_names, contexts
    # counts: each row represents a signature profile [k X 96] (dtype:torch.tensor)
    # signature_names: list of signature profiles name (dtype:list)
    # contexts: list of mutation features name (dtype:list)
'''

def get_signature_profile(b):
    # same functionality as Riccardo's but more clean
    # just read csv file as below (!!!!!index_col=0!!!!!!)
    # beta = pd.read_csv(beta_path, index_col=0)

    # list of mutation features name
    mutation_features = b.columns            # dtype:pandas.core.indexes.base.Index
    mutation_features = list(b.columns)      # dtype:list

    # list of signature profiles name
    signature_names = b.index                # dtype:pandas.core.indexes.base.Index
    signature_names = list(b.index)          # dtype:list

    # convert to torch tensor
    beta = torch.tensor(b.values)            # dtype:torch.Tensor

    return beta, signature_names, mutation_features


def get_alpha_beta(params):
    alpha = torch.exp(params["alpha"])
    alpha = alpha/(torch.sum(alpha,1).unsqueeze(-1))
    beta = torch.exp(params["beta"])
    beta = beta/(torch.sum(beta,1).unsqueeze(-1))
    return  alpha, beta

def get_alpha_beta2(a, b):
    alpha = torch.exp(a)
    alpha = alpha/(torch.sum(alpha,1).unsqueeze(-1))
    beta = torch.exp(b)
    beta = beta/(torch.sum(beta,1).unsqueeze(-1))
    return  alpha, beta



# simulate data
def catalogue(fixed_signatures, denovo_signatures):

    # read file as below
    # cosmic_catalogue = pd.read_csv(path, index_col=0)

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