import numpy as np
import pandas as pd
import torch
import pyro.distributions as dist
import csv



# ====================== DONE! ==================================
def M_csv2tensor(path):
    # input: csv file ---> output: torch.Tensor
    M_df = pd.read_csv(path)    # dtype: DataFrame
    M_np = M_df.values          # dtype: numpy.ndarray
    M = torch.tensor(M_np)      # dtype: torch.Tensor
    M = M.float()               # dtype: torch.Tensor
    return M

# ====================== DONE! ==================================
def beta_csv2tensor(path):
    # input: csv file - output: torch.Tensor
    beta_fixed_df = pd.read_csv(path, index_col=0)  # Pandas.DataFrame
    beta = beta_fixed_df.values                     # dtype: numpy.ndarray
    beta = torch.tensor(beta)                       # dtype:torch.Tensor
    beta = beta.float()
    return beta

# ====================== DONE! ==================================
def A_csv2tensor(path):
    A_df = pd.read_csv(path, header=None)           # dtype:Pandas.DataFrame
    A = torch.tensor(A_df.values)                   # dtype:torch.Tensor
    return A


# ====================== DONE! ==================================
def get_alpha_beta(params):
    alpha = torch.exp(params["alpha"])
    alpha = alpha/(torch.sum(alpha,1).unsqueeze(-1))
    beta = torch.exp(params["beta"])
    beta = beta/(torch.sum(beta,1).unsqueeze(-1))
    return  alpha, beta
    # alpha : torch.Tensor (num_samples X  k)
    # beta  : torch.Tensor (k_denovo    X  96)

# ===============================================================
def signature_names(path):
    # input: csv file - output: # list of signature profiles name
    beta_fixed_df = pd.read_csv(path, index_col=0)  # Pandas.DataFrame
    #signature_names = beta_fixed_df.index           # dtype:pandas.core.indexes.base.Index
    signature_names = list(beta_fixed_df.index)     # dtype:list
    return signature_names

# ===============================================================
def mutation_features(path):
    # input: csv file - output: list of mutation features name
    beta_fixed_df = pd.read_csv(path, index_col=0)  # Pandas.DataFrame
    #mutation_features = beta_fixed_df.columns       # dtype:pandas.core.indexes.base.Index
    mutation_features = list(beta_fixed_df.columns) # dtype:list
    return mutation_features

# ===============================================================
def M4R(path):
    df = pd.read_csv(path)
    df_T = pd.DataFrame(df.T)
    df_T.to_csv('data/data4R/M4R.csv' , header=True)

# ===============================================================
def alphas_csv2tensor(path, itr, n , k):
    # input: csv file ---> output: torch.Tensor
    alphas_df = pd.read_csv("results/alphas.csv", header=None)
    print(alphas_df)
    res = torch.zeros([itr, n, k])
    for i in range(itr):
        alphas_np = alphas_df.iloc[i*n:(i+1)*n].values  # dtype: numpy.ndarray
        alphas = torch.tensor(alphas_np)                # dtype: torch.Tensor
        alphas = alphas.float()                         # dtype: torch.Tensor
        res[i] = alphas                                 # dtype: torch.Tensor
    return res

# ===============================================================
def alphas_betas_tensor2csv(params, append):
    a, b = get_alpha_beta(params)
    alpha_np = np.array(a)
    alpha_df = pd.DataFrame(alpha_np)
    beta_np = np.array(b)
    beta_df = pd.DataFrame(beta_np)

    if (append==0):
        alpha_df.to_csv('results/alphas.csv', index=False, header=False)
        beta_df.to_csv('results/betas.csv', index=False, header=False)
    else:
        alpha_df.to_csv('results/alphas.csv', index=False, header=False, mode='a')
        beta_df.to_csv('results/betas.csv', index=False, header=False, mode='a')

# ===============================================================

def convergence(current, previous, params):
    num_samples = params["M"].size()[0]
    K_fixed = params["beta_fixed"].size()[0]
    K_denovo = params["k_denovo"]
    epsilon = params["epsilon"]
    for j in range(num_samples):
        for k in range(K_fixed + K_denovo):
            ratio = current[j][k].item() / previous[j][k].item()
            if (ratio > 1 + epsilon or ratio < 1 - epsilon ):
                #print(ratio)
                #if torch.abs(current[j][k].item() - previous[j][k]).item() > epsilon:
                return "continue"
            else:
                return "stop"


# ====================== DONE! ==================================
def M_csv4R(path):
    df = pd.read_csv(path).T
    df.to_csv('data/data4R/M4R.csv', index=True, header=True)

# ===============================================================
def beta_csv4R(path):
    df = pd.read_csv(path, index_col=0)
    df = pd.DataFrame(df.T)
    df.to_csv('data/data4R/beta4R.csv')

# ===============================================================

def generate_data(fixed_signatures, denovo_signatures):
    '''
    ====== INPUT EXAMPLE ===================================
    fixed_signatures = ["SBS1", "SBS3"]
    denovo_signatures = ["SBS5"]
    '''
    # ====== load dummy alpha ===================================
    alpha_path = "data/simulated/dummy_alpha.csv"
    df = pd.read_csv(alpha_path, header=None)   # dtype:Pandas.DataFrame
    alpha = torch.tensor(df.values)             # dtype:torch.Tensor
    alpha = alpha.float()

    # ====== load dummy theta ===================================
    theta_path = "data/simulated/dummy_theta.csv"
    df = pd.read_csv(theta_path, header=None)   # dtype:Pandas.DataFrame
    theta = df.values.tolist()[0]               # dtype:list
    #theta = torch.tensor(df.values)            # dtype:torch.Tensor

    # ====== load full beta =====================================
    beta_path = "/home/azad/Documents/thesis/SigPhylo/cosmic/cosmic_catalogue.csv"
    beta_full = pd.read_csv(beta_path, index_col=0)

    mutation_features = list(beta_full.columns) # dtype:list

    # get fixed signature profiles
    beta_fixed = beta_full.loc[fixed_signatures] # Pandas.DataFrame
    beta_fixed = beta_fixed.values          # numpy.ndarray
    beta_fixed = torch.tensor(beta_fixed)   # torch.Tensor
    beta_fixed = beta_fixed.float()         # why???????

    # get denovo signature profiles
    beta_denovo = beta_full.loc[denovo_signatures] # Pandas.DataFrame
    beta_denovo = beta_denovo.values        # numpy.ndarray
    beta_denovo = torch.tensor(beta_denovo) # torch.Tensor
    beta_denovo = beta_denovo.float()       # why?????????

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

            # sample mutation feature index for corresponding signature from categorical data
            j = dist.Categorical(b).sample().item()

            # add +1 to the mutation feature in position j in branch i
            M[i, j] += 1

    ####### export results to CSV files #############################
    with open('data/simulated/sim_catalogue.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(mutation_features)
    m_np = np.array(M)
    m_df = pd.DataFrame(m_np)
    int_df = m_df.astype(int)
    int_df.to_csv('data/simulated/sim_catalogue.csv', index=False, header=False, mode="a")

    with open('data/simulated/beta_fixed.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(mutation_features)
    fix_np = np.array(beta_fixed)
    fix_df = pd.DataFrame(fix_np)
    #fix_int_df = fix_df.astype(int)
    fix_df.to_csv('data/simulated/beta_fixed.csv', index=False, header=False, mode="a")

    with open('data/simulated/beta_denovo.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(mutation_features)
    denovo_np = np.array(beta_denovo)
    denovo_df = pd.DataFrame(denovo_np)
    #denovo_int_df = denovo_df.astype(int)
    denovo_df.to_csv('data/simulated/beta_denovo.csv', index=False, header=False, mode="a")
    #################################################################
    
    #return M, beta_fixed, beta_denovo
    # M             : dtype: torch.Tensor
    # beta_fixed    : dtype: torch.Tensor
    # beta_denovo   : dtype: torch.Tensor



'''
===================== STORAGE =====================


M = pd.read_csv("data/simulated/sim_catalogue.csv", header=None)
dummy_alpha = pd.read_csv("data/simulated/dummy_alpha.csv", header=None)

# (dtype:torch.tensor)
alpha = torch.tensor([
    [0.35, 0.50, 0.15],
    [0.52, 0.43, 0.05],
    [0.51, 0.45, 0.04],
    [0.02, 0.03, 0.95],
    [0.23, 0.46, 0.31]
    ])
theta = [1200, 3600, 2300, 1000, 1900]

def get_alpha_beta2(a, b):
    # input: torch.Tensor ---> output: torch.Tensor (enforce non-negativity and normailize)
    alpha = torch.exp(a)
    alpha = alpha/(torch.sum(alpha,1).unsqueeze(-1))
    beta = torch.exp(b)
    beta = beta/(torch.sum(beta,1).unsqueeze(-1))
    return  alpha, beta

def get_phylogeny_counts(M):
    M = M.values                                # convert to numpy array
    M = torch.tensor(np.array(M, dtype=float))  # convert to tensor
    M = M.float()
    return M

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
