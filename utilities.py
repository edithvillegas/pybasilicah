import numpy as np
import pandas as pd
import torch
import pyro.distributions as dist
import json

#------------------------ DONE! ----------------------------------
def M_read_csv(path):
    # input: csv file ---> output: torch.Tensor & mutation features (list)
    M_df = pd.read_csv(path)    # dtype: DataFrame
    M_np = M_df.values          # dtype: numpy.ndarray
    M = torch.tensor(M_np)      # dtype: torch.Tensor
    M = M.float()               # dtype: torch.Tensor
    mutation_features = list(M_df.columns) # dtype:list 
    return mutation_features, M

#------------------------ DONE! ----------------------------------
def Reconstruct_M(params):
    current_alpha, current_beta = get_alpha_beta(params)
    beta = torch.cat((params["beta_fixed"], current_beta), axis=0)
    theta = torch.sum(params["M"], axis=1)
    M_r = torch.matmul(torch.matmul(torch.diag(theta), current_alpha), beta)
    return M_r

#------------------------ DONE! ----------------------------------
def beta_read_csv(path):
    # input: csv file - output: torch.Tensor & signature names (list) & mutation features (list)
    beta_df = pd.read_csv(path, index_col=0)  # Pandas.DataFrame
    beta = beta_df.values                     # dtype: numpy.ndarray
    beta = torch.tensor(beta)                       # dtype:torch.Tensor
    beta = beta.float()

    signature_names = list(beta_df.index)     # dtype:list
    mutation_features = list(beta_df.columns) # dtype:list

    return signature_names, mutation_features, beta

#------------------------ DONE! ----------------------------------
def beta_read_name(beta_name_list):
    # input: list - output: torch.Tensor & signature names (list) & mutation features (list)
    beta_path = "/home/azad/Documents/thesis/SigPhylo/cosmic/cosmic_catalogue.csv"
    beta_full = pd.read_csv(beta_path, index_col=0)

    beta_df = beta_full.loc[beta_name_list]   # Pandas.DataFrame
    beta = beta_df.values               # numpy.ndarray
    beta = torch.tensor(beta)           # torch.Tensor
    beta = beta.float()                 # why???????

    signature_names = list(beta_df.index)     # dtype:list
    mutation_features = list(beta_df.columns) # dtype:list

    return signature_names, mutation_features, beta

#------------------------ DONE! ----------------------------------
def A_read_csv(path):
    A_df = pd.read_csv(path, header=None)           # dtype:Pandas.DataFrame
    A = torch.tensor(A_df.values)                   # dtype:torch.Tensor
    return A

#------------------------ DONE! ----------------------------------
def get_alpha_beta(params):
    alpha = torch.exp(params["alpha"])
    alpha = alpha/(torch.sum(alpha, 1).unsqueeze(-1))
    beta = torch.exp(params["beta"])
    beta = beta/(torch.sum(beta,1).unsqueeze(-1))
    return  alpha, beta
    # alpha : torch.Tensor (num_samples X  k)
    # beta  : torch.Tensor (k_denovo    X  96)

# ====================== DONE! ==================================
def alpha_batch_df(df, alpha):
    # df    :   pandas.DataFrame
    # alpha :   torch.Tensor
    alpha_flat = torch.flatten(alpha)
    alpha_numpy = np.array(alpha_flat)
    alpha_series = pd.Series(alpha_numpy)
    df = df.append(alpha_series, ignore_index=True)
    return df

#------------------------ DONE! ----------------------------------
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#------------------------ DONE! ----------------------------------
def likelihood(params):
    alpha, beta_denovo = get_alpha_beta(params)
    theta = torch.sum(params["M"], axis=1)
    beta = torch.cat((params["beta_fixed"], beta_denovo), axis=0)
    LH_Matrix = dist.Poisson(
        torch.matmul(
            torch.matmul(torch.diag(theta), alpha), 
            beta)
            ).log_prob(params["M"])
    LH = torch.sum(LH_Matrix)
    LH = float("{:.3f}".format(LH.item()))
    #p = params["k_denovo"]*96 + params["M"].shape[0] * (params["k_denovo"] + params["beta_fixed"].shape[0])
    #BIC = p*torch.log(torch.tensor(params["M"].shape[0]*params["M"].shape[1])) -2*LH
    return LH

#------------------------ DONE! ----------------------------------
def best(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    max = -10000000
    ind = -1
    for key, value in data["output"].items():
        tmp = value["likelihoods"][-1]
        if tmp > max:
            max = tmp
            ind = key
    best_k = data["output"][ind]["k_denovo"]
    best_lambda = data["output"][ind]["lambda"]

    if ind == -1:
        return "FALSE", "FALSE"

    return best_k, best_lambda

# ====================== DONE! ==================================
def convergence(current_alpha, previous_alpha, params):
    num_samples = params["M"].size()[0]
    K_fixed = params["beta_fixed"].size()[0]
    K_denovo = params["k_denovo"]
    epsilon = params["epsilon"]
    for j in range(num_samples):
        for k in range(K_fixed + K_denovo):
            ratio = current_alpha[j][k].item() / previous_alpha[j][k].item()
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

    #------- get beta fixed & denovo ----------------------------
    signature_names, mutation_features, beta_fixed = beta_read_name(fixed_signatures)
    signature_names, mutation_features, beta_denovo = beta_read_name(denovo_signatures)
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


'''
def cosine_similarity(M, M_r):
    cos = []
    for i in range(M.size()[0]):
        c = (torch.dot(M[i], M_r[i]) / (torch.norm(M[i]) * torch.norm(M_r[i]))).item()
        value = float("{:.3f}".format(c))
        cos.append(value)
    
    #r = sum(i > threshold for i in cos) / len(cos)
    return cos


#df = pd.DataFrame(columns=["k_denovo", "lambda", "LH"])
k = value["k_denovo"]  # int
landa = value["lambda"]    # int
L = value["likelihoods"][-1]   # float
x_numpy = np.array([k, landa, L])
x_series = pd.Series(x_numpy, index=["k_denovo", "lambda", "LH"])
df = df.append(x_series, ignore_index=True)

k_max = df.iloc[df['LH'].idxmax()]["k_denovo"]
lambda_max = df.iloc[df['LH'].idxmax()]["lambda"]
'''