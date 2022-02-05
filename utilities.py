import numpy as np
import pandas as pd
import torch
import pyro.distributions as dist
import json


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
def beta_list2tensor(beta_list_path):
    with open(beta_list_path) as f:
            lines = f.read()
    fixed_signatures = lines.splitlines()[0].split(sep=",")

    beta_path = "/home/azad/Documents/thesis/SigPhylo/cosmic/cosmic_catalogue.csv"
    beta_full = pd.read_csv(beta_path, index_col=0)

    beta_fixed = beta_full.loc[fixed_signatures] # Pandas.DataFrame
    beta_fixed = beta_fixed.values          # numpy.ndarray
    beta_fixed = torch.tensor(beta_fixed)   # torch.Tensor
    beta_fixed = beta_fixed.float()         # why???????

    return beta_fixed

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

# ====================== DONE! ==================================
def signature_names(path):
    # input: csv file - output: # list of signature profiles name
    beta_fixed_df = pd.read_csv(path, index_col=0)  # Pandas.DataFrame
    #signature_names = beta_fixed_df.index          # dtype:pandas.core.indexes.base.Index
    signature_names = list(beta_fixed_df.index)     # dtype:list
    return signature_names

# ====================== DONE! ==================================
def beta_mutation_features(path):
    # input: csv file - output: list of mutation features name
    df = pd.read_csv(path, index_col=0)  # Pandas.DataFrame
    #mutation_features = beta_fixed_df.columns       # dtype:pandas.core.indexes.base.Index
    mutation_features = list(df.columns) # dtype:list
    return mutation_features

# ====================== DONE! ==================================
def M_mutation_features(path):
    # input: csv file - output: list of mutation features name
    df = pd.read_csv(path, index_col=None)  # Pandas.DataFrame
    #mutation_features = beta_fixed_df.columns       # dtype:pandas.core.indexes.base.Index
    mutation_features = list(df.columns) # dtype:list
    return mutation_features

# ====================== DONE! ==================================
def alpha_batch_df(df, alpha):
    # df    :   pandas.DataFrame
    # alpha :   torch.Tensor
    alpha_flat = torch.flatten(alpha)
    alpha_numpy = np.array(alpha_flat)
    alpha_series = pd.Series(alpha_numpy)
    df = df.append(alpha_series, ignore_index=True)
    return df

# ====================== DONE! ==================================
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# ===============================================================
def likelihoods(params, likelihoods):
    alpha, beta_denovo = get_alpha_beta(params)
    theta = torch.sum(params["M"], axis=1)
    beta = torch.cat((params["beta_fixed"], beta_denovo), axis=0)
    likelihood_matrix = dist.Poisson(
        torch.matmul(
            torch.matmul(torch.diag(theta), alpha), beta)).log_prob(params["M"])
    likelihood = torch.sum(likelihood_matrix)
    value = float("{:.3f}".format(likelihood.item()))
    likelihoods.append(value)

    return likelihoods

# ====================== DONE! ==================================????????
def cosine_sim(M, M_r):
    #print("hello world")
    #M = utilities.M_csv2tensor("/home/azad/Documents/thesis/SigPhylo/data/simulated/data_sigphylo.csv")
    #M_r = utilities.M_csv2tensor("/home/azad/Documents/thesis/SigPhylo/data/results/KL/K_1_L_0/M_r.csv")
    num_samples = M.size()[0]

    cos = []
    for i in range(num_samples):
            c = (torch.dot(M[i], M_r[i]) / (torch.norm(M[i])*torch.norm(M_r[i]))).item()
            value = float("{:.3f}".format(c))
            cos.append(value)
    
    #r = sum(i > threshold for i in cos) / len(cos)
    return cos

# ====================== DONE! ==================================
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
                #if torch.abs(current[j][k].item() - previous[j][k]).item()) > epsilon:
                return "continue"
            else:
                return "stop"

# ====================== DONE! ==================================
def generate_data():
    beta_list_path = "data/simulated/beta_list.txt" # load beta list
    with open(beta_list_path) as f:
        lines = f.read()
    fixed_signatures = lines.splitlines()[0].split(sep=",")
    denovo_signatures = lines.splitlines()[1].split(sep=",")

    # ====== load expected alpha ===================================
    alpha_path = "data/simulated/expected_alpha.csv"
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

    # ====== create mutation features list
    mutation_features = list(beta_full.columns) # dtype:list

    # ====== get fixed signature profiles
    beta_fixed = beta_full.loc[fixed_signatures] # Pandas.DataFrame
    beta_fixed = beta_fixed.values          # numpy.ndarray
    beta_fixed = torch.tensor(beta_fixed)   # torch.Tensor
    beta_fixed = beta_fixed.float()         # why???????

    # ====== get denovo signature profiles
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
        p = alpha[i]    # selecting branch i
        for k in range(theta[i]):   # iterate for number of the mutations in branch i

            # sample signature profile index from categorical data
            b = beta[dist.Categorical(p).sample().item()]

            # sample mutation feature index for corresponding signature from categorical data
            j = dist.Categorical(b).sample().item()

            # add +1 to the mutation feature in position j in branch i
            M[i, j] += 1

    # ======= export results to CSV files =============================

    # mutational catalogue
    m_np = np.array(M)
    m_df = pd.DataFrame(m_np, columns=mutation_features)
    int_df = m_df.astype(int)
    int_df.to_csv('data/simulated/data_sigphylo.csv', index=False, header=True)

    # beta fixed
    fix_np = np.array(beta_fixed)
    fix_df = pd.DataFrame(fix_np, index=fixed_signatures, columns=mutation_features)
    fix_df.to_csv('data/simulated/beta_fixed.csv', index=True, header=True)

    # beta denovo
    denovo_np = np.array(beta_denovo)
    denovo_df = pd.DataFrame(denovo_np, index=denovo_signatures, columns=mutation_features)
    denovo_df.to_csv('data/simulated/beta_denovo.csv', index=True, header=True)

