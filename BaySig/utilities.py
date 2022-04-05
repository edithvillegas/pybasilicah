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
========= Reading mutational catalogue from csv file =============
==================================================================
M_df = pd.read_csv(M_path)                          dtype:DataFrame
M_tensor = torch.tensor(M_df.values).float()        dtype:torch.Tensor
mutation_features = list(M_df.columns)              dtype:list
==================================================================
========= Reading signature profiles from csv file ===============
==================================================================
beta_df = pd.read_csv(path, index_col=0)            dtype:DataFrame
beta_tensor = torch.tensor(beta_df.values).float()  dtype:torch.Tensor
signature_names = list(beta_df.index)               dtype:list
mutation_features = list(beta_df.columns)           dtype:list
------------------------------------------------------------------
'''

#------------------------ DONE! ----------------------------------[PASSED]
def Reconstruct_M(params):
    # output --- dtype: tensor
    alpha, beta_denovo = get_alpha_beta(params)
    beta = torch.cat((params["beta_fixed"], beta_denovo), axis=0)
    theta = torch.sum(params["M"], axis=1)
    M_r = torch.matmul(torch.matmul(torch.diag(theta), alpha), beta)
    return M_r

#------------------------ DONE! ----------------------------------[PASSED]
def beta_read_name(beta_name_list, cosmic_path):
    # input: (list, string) - output: torch.Tensor & signature names (list) & mutation features (list)
    # input: (list, string) - output: dataframe (not ok)
    beta_full = pd.read_csv(cosmic_path, index_col=0)

    beta_df = beta_full.loc[beta_name_list]   # Pandas.DataFrame
    #beta = beta_df.values               # numpy.ndarray
    #beta = torch.tensor(beta)           # torch.Tensor
    #beta = beta.float()                 # why???????

    #signature_names = list(beta_df.index)     # dtype:list
    #mutation_features = list(beta_df.columns) # dtype:list

    # beta denovo
    #beta_np = np.array(beta)
    #beta_df = pd.DataFrame(beta_np, index=signature_names, columns=mutation_features)

    #return signature_names, mutation_features, beta
    return beta_df

#------------------------ DONE! ----------------------------------[PASSESD]
def get_alpha_beta(params):
    alpha = torch.exp(params["alpha"])
    alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))
    beta = torch.exp(params["beta"])
    beta = beta / (torch.sum(beta,1).unsqueeze(-1))
    return  alpha, beta
    # alpha : torch.Tensor (num_samples X  k)
    # beta  : torch.Tensor (k_denovo    X  96)

#------------------------ DONE! ----------------------------------
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#------------------------ DONE! ----------------------------------[PASSED]
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

#------------------------ DONE! ----------------------------------[PASSED]
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


#------------------------ DONE! ----------------------------------[PASSED]
def target_generator(num_samples, k_fixed, k_denovo, cosmic_path):

    #signature_names, mutation_features, _ = beta_read_csv(cosmic_path)
    beta_df = pd.read_csv(cosmic_path, index_col=0)
    signature_names = list(beta_df.index)
    mutation_features = list(beta_df.columns)

    #------- alpha ----------------------------------------------
    matrix = np.random.rand(num_samples, k_fixed + k_denovo)
    alpha_tensor = torch.tensor(matrix / matrix.sum(axis=1)[:,None]).float()

    #------- beta -----------------------------------------------
    fixed_signatures = random.sample(signature_names, k=k_fixed)
    for item in fixed_signatures:
        signature_names.remove(item)
    denovo_signatures = random.sample(signature_names, k=k_denovo)

    beta_fixed_df = beta_read_name(fixed_signatures, cosmic_path)
    beta_fixed_tensor = torch.tensor(beta_fixed_df.values).float()

    beta_denovo_df = beta_read_name(denovo_signatures, cosmic_path)
    beta_denovo_tensor = torch.tensor(beta_denovo_df.values).float()
    
    beta = torch.cat((beta_fixed_tensor, beta_denovo_tensor), axis=0)

    #------- theta ----------------------------------------------
    theta = random.sample(range(1000, 4000), k=num_samples)

    
    #------- check dimensions -----------------------------------
    m_alpha = alpha_tensor.size()[0]    # no. of branches (alpha)
    m_theta = len(theta)                # no. of branches (theta)
    k_alpha = alpha_tensor.size()[1]    # no. of signatures (alpha)
    k_beta = beta.size()[0]             # no. of signatures (beta)
    
    if not(m_alpha == m_theta and k_alpha == k_beta):
        print("WRONG INPUT!")
        return 10
    
    #num_samples = alpha_tensor.size()[0]       # number of branches
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

    # ======= convert to DataFrame ====================================

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

    # return all in dataframe format
    return M, alpha, beta_fixed, beta_denovo


#------------------------ DONE! ----------------------------------[PASSED]
def input_generator(num_samples, k_fixed, k_denovo, cosmic_path):

    # TARGET DATA -----------------------------------------------------------------------------
    # create random simulated target data (all in dataframe format)
    M, alpha, beta_fixed, beta_denovo = target_generator(num_samples, k_fixed, k_denovo, cosmic_path)

    # TEST DATA -------------------------------------------------------------------------------
    # create random simulated test fixed signature

    # full cosmic signatures names (dtype: list)
    #exc_cosmic_names, _, _ = beta_read_csv(cosmic_path)
    cosmic_df = pd.read_csv(cosmic_path, index_col=0)
    exc_cosmic_names = cosmic_df.columns

    # full target beta signatures names (dtype: list)
    beta_names = list(beta_fixed.index) + list(beta_denovo.index)

    # create cosmic signatures list excluded target signatures (fixed + denovo)
    for signature in beta_names:
        exc_cosmic_names.remove(signature)
    
    overlap = random.randint(0, k_fixed)    # no. of common fixed signatures (target intersect test)
    exceed = random.randint(0, k_fixed)     # no. of different fixed signatures (test - target)

    overlap_sigs = random.sample(list(beta_fixed.index), k=overlap)  # common fixed signatures list
    exceed_sigs = random.sample(exc_cosmic_names, k=exceed)          # different fixed signatures list

    beta_fixed_test_names, mutation_features, beta_fixed_test_tensor = beta_read_name(overlap_sigs + exceed_sigs, cosmic_path)
    fix_test_np = np.array(beta_fixed_test_tensor)
    beta_fixed_test = pd.DataFrame(fix_test_np, index=beta_fixed_test_names, columns=mutation_features)

    data = {
        "M" : M,                                            # dataframe
        "alpha" : alpha,                                    # dataframe
        "beta_fixed" : beta_fixed,                          # dataframe
        "beta_denovo" : beta_denovo,                        # dataframe
        "overlap" : overlap,                                # int
        "exceed" : exceed,                                  # int
        "beta_fixed_test" : beta_fixed_test                 # dataframe
    }

    return data


#------------------------ DONE! ----------------------------------[PASSED]
# note: just need a liitle bit code optimization
def fixedFilter(alpha_inferred, beta_fixed_test):
    # alpha_inferred -------------------- dtype: torch.Tensor
    # beta_fixed_test ------------------- dtype: data.frame
    # output ---> new beta_fixed_test --- dtype: list

    beta_fixed_list = list(beta_fixed_test.index)
    k_fixed = len(beta_fixed_list)
    a = (torch.sum(alpha_inferred, axis=0) / np.array(alpha_inferred).shape[0]).tolist()[:k_fixed]
    b = [x for x in a if x <= 0.05]
    
    excluded = []
    if len(b)==0:
        #print("all signatures are significant!")
        return beta_fixed_list
    else:
        for i in b:
            index = a.index(i)
            excluded.append(beta_fixed_list[index])
            print("Signature", beta_fixed_list[index], "is not included!")
    
    for j in excluded:
        beta_fixed_list.remove(j)
    
    return beta_fixed_list  # dtype: list


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
        
    return match


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

