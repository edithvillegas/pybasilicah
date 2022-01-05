import torch
import numpy as np

# load data

# input : dataframe - output : tensor
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

