import numpy as np
import pandas as pd
import SigPhylo
import torch
import pyro.distributions as dist
import json
import copy
import multiprocessing as mp

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

#------------------------ DONE! ---------------------------------- Not used anymore!
def alpha_read_csv(path):
    # input: csv file - output: torch.Tensor & signature names (list)
    alpha_df = pd.read_csv(path, header=None)  # Pandas.DataFrame
    alpha = alpha_df.values                     # dtype: numpy.ndarray
    alpha = torch.tensor(alpha)                 # dtype:torch.Tensor
    alpha = alpha.float()
    #signature_names = list(alpha_df.index)     # dtype:list

    return alpha

#------------------------ DONE! ----------------------------------
def beta_read_csv(path):
    # input: csv file - output: torch.Tensor & signature names (list) & mutation features (list)
    beta_df = pd.read_csv(path, index_col=0)  # Pandas.DataFrame
    beta = beta_df.values                     # dtype: numpy.ndarray
    beta = torch.tensor(beta)                 # dtype:torch.Tensor
    beta = beta.float()

    signature_names = list(beta_df.index)     # dtype:list
    mutation_features = list(beta_df.columns) # dtype:list

    return signature_names, mutation_features, beta

#------------------------ DONE! ----------------------------------
def beta_read_name(beta_name_list, cosmic_path):
    # input: list - output: torch.Tensor & signature names (list) & mutation features (list)
    beta_full = pd.read_csv(cosmic_path, index_col=0)

    beta_df = beta_full.loc[beta_name_list]   # Pandas.DataFrame
    beta = beta_df.values               # numpy.ndarray
    beta = torch.tensor(beta)           # torch.Tensor
    beta = beta.float()                 # why???????

    signature_names = list(beta_df.index)     # dtype:list
    mutation_features = list(beta_df.columns) # dtype:list

    return signature_names, mutation_features, beta

#------------------------ DONE! ----------------------------------
def names_to_beta(fixed, denovo, cosmic_path):
    # input: list  | output: full beta tensor
    # e.g., full_beta(["SBS5"], ["SBS84", "SBS45"])
    fixed_signature_names, mutation_features, beta_fixed_tensor = beta_read_name(fixed, cosmic_path)
    denovo_signature_names, mutation_features, beta_denovo_tensor = beta_read_name(denovo, cosmic_path)
    beta = torch.cat((beta_fixed_tensor, beta_denovo_tensor), axis=0)
    return beta


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

#------------------------ DONE! ----------------------------------
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#------------------------ DONE! ----------------------------------
def log_likelihood(params):
    alpha, beta_denovo = get_alpha_beta(params)
    theta = torch.sum(params["M"], axis=1)
    beta = torch.cat((params["beta_fixed"], beta_denovo), axis=0)
    log_L_Matrix = dist.Poisson(
        torch.matmul(
            torch.matmul(torch.diag(theta), alpha), 
            beta)
            ).log_prob(params["M"])
    log_L = torch.sum(log_L_Matrix)
    log_L = float("{:.3f}".format(log_L.item()))

    return log_L

def BIC(params):
    alpha, beta_denovo = get_alpha_beta(params)
    theta = torch.sum(params["M"], axis=1)
    beta = torch.cat((params["beta_fixed"], beta_denovo), axis=0)
    log_L_Matrix = dist.Poisson(
        torch.matmul(
            torch.matmul(torch.diag(theta), alpha), 
            beta)
            ).log_prob(params["M"])
    log_L = torch.sum(log_L_Matrix)
    log_L = float("{:.3f}".format(log_L.item()))

    k = (params["M"].shape[0] * params["k_denovo"]) + (params["k_denovo"] * params["M"].shape[1])
    n = params["M"].shape[0] * params["M"].shape[1]
    bic = k * torch.log(torch.tensor(n)) - (2 * log_L)
    return bic.item()


#------------------------ DONE! ----------------------------------
def generate_data(argus, cosmic_path):

    alpha_tensor = argus["alpha"]
    fixed_signatures = argus["fixed_signatures"]
    denovo_signatures = argus["denovo_signatures"]
    theta = argus["theta"]
    A_tensor = argus["A_tensor"]

    #------- alpha ----------------------------------------------
    '''
    alpha_tensor = torch.tensor(
        [[0.95, 0.05], 
        [0.40, 0.60], 
        [0.04, 0.96]]
        )
    '''
    #------- beta -----------------------------------------------
    '''
    fixed_signatures = ["SBS5"]
    denovo_signatures = ["SBS84"]
    '''
    signature_names, mutation_features, beta_fixed_tensor = beta_read_name(fixed_signatures, cosmic_path)
    signature_names, mutation_features, beta_denovo_tensor = beta_read_name(denovo_signatures, cosmic_path)
    beta = torch.cat((beta_fixed_tensor, beta_denovo_tensor), axis=0)

    #------- theta ----------------------------------------------
    '''
    theta = [1200, 3600, 2300]
    '''

    #------- A --------------------------------------------------
    '''
    A_tensor = torch.tensor(
        [[1,1,1], 
        [1,1,1], 
        [1,1,1]])
    '''

    #------- check dimensions -----------------------------------
    m_alpha = alpha_tensor.size()[0]    # no. of branches
    k_alpha = alpha_tensor.size()[1]    # no. of signatures
    k_beta = beta.size()[0]             # no. of signatures
    m_theta = len(theta)
    nrow_A = A_tensor.size()[0]
    ncol_A = A_tensor.size()[1]
    if not(m_alpha == m_theta == nrow_A ==ncol_A and k_alpha == k_beta):
        print("WRONG INPUT!")
        return 10
    
    num_samples = alpha_tensor.size()[0]   # number of branches
    M_tensor = torch.zeros([num_samples, 96])   # initialize mutational catalogue with zeros

    for i in range(num_samples):
        p = alpha_tensor[i]    # selecting branch i
        for k in range(theta[i]):   # iterate for number of the mutations in branch i

            # sample signature profile index from categorical data
            b = beta[dist.Categorical(p).sample().item()]

            # sample mutation feature index for corresponding signature from categorical data
            j = dist.Categorical(b).sample().item()

            # add +1 to the mutation feature in position j in branch i
            M_tensor[i, j] += 1

    # ======= export results to CSV files =============================

    # phylogeny
    m_np = np.array(M_tensor)
    m_df = pd.DataFrame(m_np, columns=mutation_features)
    M = m_df.astype(int)

    # alpha
    alpha_columns = fixed_signatures + denovo_signatures
    alpha_np = np.array(alpha_tensor)
    alpha = pd.DataFrame(alpha_np, columns=alpha_columns)

    # beta fixed
    fix_np = np.array(beta_fixed_tensor)
    beta_fixed = pd.DataFrame(fix_np, index=fixed_signatures, columns=mutation_features)
    
    # beta denovo
    denovo_np = np.array(beta_denovo_tensor)
    beta_denovo = pd.DataFrame(denovo_np, index=denovo_signatures, columns=mutation_features)

    # A
    A_np = np.array(A_tensor)
    A = pd.DataFrame(A_np)

    # return all in dataframe format
    return M, alpha, beta_fixed, beta_denovo, A



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
                #if torch.abs(current[j][k].item() - previous[j][k]).item()) > epsilon:
                return "continue"
    return "stop"


#=========================================================================================
#======================== Single & Parallel Running ======================================
#=========================================================================================

def make_args(params, k_list, lambda_list):
    args_list = []
    for k in k_list:
        for landa in lambda_list:
            b = copy.deepcopy(params)
            b["k_denovo"] = k
            b["lambda"] = landa
            args_list.append(b)
    return args_list


def singleProcess(params, k_list, lambda_list):
    output_data = {}    # initialize output data dictionary
    i = 1
    for k in k_list:
        for landa in lambda_list:
            #print("k_denovo =", k, "| lambda =", landa)

            params["k_denovo"] = k
            params["lambda"] = landa

            output_data[str(i)] = SigPhylo.single_run(params)
            i += 1
    return output_data


def multiProcess(params, k_list, lambda_list):
    args_list = make_args(params, k_list, lambda_list)
    output_data = {}
    with mp.Pool(processes=mp.cpu_count()) as pool_obj:
        results = pool_obj.map(SigPhylo.single_run, args_list)
    
    for i in range(len(results)):
        output_data[str(i+1)] = results[i]
    return output_data

