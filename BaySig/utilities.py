import numpy as np
import pandas as pd
import torch
import pyro.distributions as dist
import json
import copy
import multiprocessing as mp
import torch.nn.functional as F
import random
import run

'''
INSTRUCTIONS:

==================================================================
========= Mutational catalogue Read by CSV =======================
==================================================================
M_df = pd.read_csv(M_path)                          dtype:DataFrame
M_tensor = torch.tensor(M_df.values).float()        dtype:torch.Tensor
mutation_features = list(M_df.columns)              dtype:list
==================================================================
========= beta Read by CSV =======================================
==================================================================
beta_df = pd.read_csv(path, index_col=0)            dtype:DataFrame
beta_tensor = torch.tensor(beta_df.values).float()  dtype:torch.Tensor
signature_names = list(beta_df.index)               dtype:list
mutation_features = list(beta_df.columns)           dtype:list
==================================================================
========= beta Read by name ======================================
==================================================================
cosmic_df = pd.read_csv(cosmic_path, index_col=0)
beta_df = cosmic_df.loc[beta_name_list]             dtype:DataFrame
beta_tensor = torch.tensor(beta_df.values).float()  dtype:torch.Tensor
signature_names = list(beta_df.index)               dtype:list
mutation_features = list(beta_df.columns)           dtype:list
------------------------------------------------------------------
'''

#-----------------------------------------------------------------[PASSED]
def Reconstruct_M(params):
    # output --- dtype: tensor
    alpha, beta_denovo = get_alpha_beta(params)
    beta = torch.cat((params["beta_fixed"], beta_denovo), axis=0)
    theta = torch.sum(params["M"], axis=1)
    M_r = torch.matmul(torch.matmul(torch.diag(theta), alpha), beta)
    return M_r

#-----------------------------------------------------------------[PASSESD]
def get_alpha_beta(params):
    alpha = torch.exp(params["alpha"])
    alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))
    beta = torch.exp(params["beta"])
    beta = beta / (torch.sum(beta, 1).unsqueeze(-1))
    return  alpha, beta
    # alpha : torch.Tensor (num_samples X  k)
    # beta  : torch.Tensor (k_denovo    X  96)

#------------------------ DONE! ----------------------------------
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#-----------------------------------------------------------------[PASSED]
def log_likelihood(params):
    alpha, beta_denovo = get_alpha_beta(params)
    theta = torch.sum(params["M"], axis=1)
    beta = torch.cat((params["beta_fixed"], beta_denovo), axis=0)
    log_likelihood_Matrix = dist.Poisson(
        torch.matmul(
            torch.matmul(torch.diag(theta), alpha), 
            beta)
            ).log_prob(params["M"])
    log_like = torch.sum(log_likelihood_Matrix)
    log_like = float("{:.3f}".format(log_like.item()))

    return log_like

#-----------------------------------------------------------------[PASSED]
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

    k = (params["M"].shape[0] * (params["k_denovo"] + params["beta_fixed"].shape[0])) + (params["k_denovo"] * params["M"].shape[1])
    n = params["M"].shape[0] * params["M"].shape[1]
    bic = k * torch.log(torch.tensor(n)) - (2 * log_L)
    return bic.item()


#-----------------------------------------------------------------[PASSED]
def target_generator(num_samples, num_sig, cosmic_path):

    cosmic_df = pd.read_csv(cosmic_path, index_col=0)
    signature_names = list(cosmic_df.index)
    mutation_features = list(cosmic_df.columns)

    # random signature selection from cosmic
    signatures = random.sample(signature_names, k=num_sig)

    #------- alpha ----------------------------------------------
    matrix = np.random.rand(num_samples, num_sig)
    alpha_tensor = torch.tensor(matrix / matrix.sum(axis=1)[:, None]).float()
    alpha_np = np.array(alpha_tensor)
    alpha_df = pd.DataFrame(alpha_np, columns=signatures)

    #------- beta -----------------------------------------------
    beta_df = cosmic_df.loc[signatures]
    beta_tensor = torch.tensor(beta_df.values).float()

    #------- theta ----------------------------------------------
    theta = random.sample(range(1000, 4000), k=num_samples)
    
    #------- check dimensions -----------------------------------
    m_alpha = alpha_tensor.size()[0]    # no. of branches (alpha)
    m_theta = len(theta)                # no. of branches (theta)
    k_alpha = alpha_tensor.size()[1]    # no. of signatures (alpha)
    k_beta = beta_tensor.size()[0]      # no. of signatures (beta)
    if not(m_alpha == m_theta and k_alpha == k_beta):
        print("WRONG INPUT!")
        return 10
    
    M_tensor = torch.zeros([num_samples, 96])   # initialize mutational catalogue with zeros

    for i in range(num_samples):
        p = alpha_tensor[i]         # selecting branch i
        for k in range(theta[i]):   # iterate for number of the mutations in branch i

            # sample signature profile index from categorical data
            b = beta_tensor[dist.Categorical(p).sample().item()]

            # sample mutation feature index for corresponding signature from categorical data
            j = dist.Categorical(b).sample().item()

            # add +1 to the mutation feature in position j in branch i
            M_tensor[i, j] += 1

    # ======= convert to DataFrame ====================================

    # phylogeny
    m_np = np.array(M_tensor)
    m_df = pd.DataFrame(m_np, columns=mutation_features)
    M_df = m_df.astype(int)

    # return all in dataframe format
    return M_df, alpha_df, beta_df


#-----------------------------------------------------------------[PASSED]
def input_generator(num_samples, num_sig, cosmic_path):

    # TARGET DATA -----------------------------------------------------------------------------
    M_df, alpha_df, beta_df = target_generator(num_samples, num_sig, cosmic_path)
    beta_names = list(beta_df.index)                    # target beta signatures names (dtype: list)

    # TEST DATA -------------------------------------------------------------------------------
    cosmic_df = pd.read_csv(cosmic_path, index_col=0)   # cosmic signatures (dtype: data.Frame)
    cosmic_names = list(cosmic_df.index)                # cosmic signatures names (dtype: list)

    # exclude beta target signatures from cosmic
    for signature in beta_names:
        cosmic_names.remove(signature)
    
    overlap = 0
    while overlap == 0:
        overlap = random.randint(0, num_sig)    # common fixed signatures (target intersect test)
    extra = random.randint(0, num_sig)         # different fixed signatures (test minus target)

    overlap_sigs = random.sample(beta_names, k=overlap)     # common fixed signatures list
    extra_sigs = random.sample(cosmic_names, k=extra)       # different fixed signatures list

    beta_fixed_test = cosmic_df.loc[overlap_sigs + extra_sigs]

    data = {
        "M" : M_df,                                         # dataframe
        "alpha" : alpha_df,                                 # dataframe
        "beta" : beta_df,                                   # dataframe
        "beta_fixed_test" : beta_fixed_test,                # dataframe
        "overlap" : overlap,                                # int
        "extra" : extra,                                    # int
    }

    return data


#-----------------------------------------------------------------[PASSED]
def fixedFilter(alpha_tensor, beta_df, theta_np):
    # alpha_tensor (inferred alpha) ------------------------- dtype: torch.Tensor
    # beta_df (input beta fixed) ---------------------------- dtype: data.frame
    # theta_np ---------------------------------------------- dtype: numpy

    beta_test_list = list(beta_df.index)
    #k_fixed = len(beta_test_list)

    alpha = np.array(alpha_tensor)
    theta = np.expand_dims(theta_np, axis = 1)
    x = np.multiply(alpha, theta)
    a = np.sum(x, axis=0) / np.sum(x)
    b = [x for x in a if x <= 0.05]

    #a = (torch.sum(alpha_inf, axis=0) / np.array(alpha_inf).shape[0]).tolist()[:k_fixed]
    #b = [x for x in a if x <= 0.05]

    a = list(a)
    b = list(b)
    
    excluded = []
    if len(b)==0:
        #print("all signatures are significant!")
        return beta_test_list
    else:
        for j in b:
            index = a.index(j)
            if index < len(beta_test_list):
                excluded.append(beta_test_list[index])
            #print("Signature", beta_test_list[index], "is not included!")
    
    for k in excluded:
        beta_test_list.remove(k)
    
    # list of significant signatures in beta_test ---- dtype: list
    return beta_test_list  # dtype: list

#-----------------------------------------------------------------[PASSED]
def denovoFilter(beta_inferred, cosmic_path):
    # beta_inferred -- dtype: tensor
    # cosmic_path ---- dtype: string
    cosmic_df = pd.read_csv(cosmic_path, index_col=0)
    match = []
    for index in range(beta_inferred.size()[0]):
        denovo = beta_inferred[index]   # dtype: tensor
        denovo = denovo[None, :]        # dtype: tensor (convert from 1D to 2D)
        maxScore = 0
        cosMatch = ""
        for cosName in list(cosmic_df.index):
            cos = cosmic_df.loc[cosName]                    # pandas Series
            cos_tensor = torch.tensor(cos.values).float()   # dtype: tensor
            cos_tensor = cos_tensor[None, :]                # dtype: tensor (convert from 1D to 2D)

            #score = F.kl_div(denovo, cos_tensor, reduction="batchmean").item()
            score = F.cosine_similarity(denovo, cos_tensor).item()
            if score >= maxScore:
                maxScore = score
                cosMatch = cosName
        if maxScore > 0.9:
            match.append(cosMatch)
        
    return match    # dtype: list


def stopRun(new_list, old_list, denovo_list):
    new_list.sort()
    old_list.sort()
    if new_list==old_list and len(denovo_list)==0:
        return True
    else:
        return False


'''
#=========================================================================================
#======================== Single & Parallel Running ======================================
#=========================================================================================

def make_args(params, k_list):
    args_list = []
    for k in k_list:
        b = copy.deepcopy(params)
        b["k_denovo"] = k
        args_list.append(b)
    return args_list


def singleProcess(params, k_list):
    output_data = {}    # initialize output data dictionary
    i = 1
    for k in k_list:
        #print("k_denovo =", k)

        params["k_denovo"] = k

        output_data[str(i)] = run.single_run(params)
        i += 1
    return output_data


def multiProcess(params, k_list):
    args_list = make_args(params, k_list)
    output_data = {}
    with mp.Pool(processes=mp.cpu_count()) as pool_obj:
        results = pool_obj.map(run.single_run, args_list)
    
    for i in range(len(results)):
        output_data[str(i+1)] = results[i]
    return output_data
'''

#------------------------ DONE! ----------------------------------[passed]
# note: just check the order of kl-divergence arguments and why the value is negative
def regularizer(beta_fixed, beta_denovo):
    loss = 0
    for fixed in beta_fixed:
        for denovo in beta_denovo:
            loss += F.kl_div(fixed, denovo, reduction="batchmean").item()

    return loss

#------------------------ DONE! ----------------------------------[passed]
def custom_likelihood(alpha, beta_fixed, beta_denovo, M):
    # build full signature profile (beta) matrix
    beta = torch.cat((beta_fixed, beta_denovo), axis=0)
    likelihood =  dist.Poisson(torch.matmul(torch.matmul(torch.diag(torch.sum(M, axis=1)), alpha), beta)).log_prob(M)
    regularization = regularizer(beta_fixed, beta_denovo)
    return likelihood + regularization

