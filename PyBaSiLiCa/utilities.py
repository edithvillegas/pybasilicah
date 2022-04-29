import numpy as np
import pandas as pd
import torch
import pyro.distributions as dist
import json
import multiprocessing as mp
import torch.nn.functional as F
import random
import basilica

'''
======================================================================================================
************************************* CODING TIPS ****************************************************
======================================================================================================

========= Mutational catalogue Read by CSV =======================

M_df = pd.read_csv(M_path)                          dtype:DataFrame
M_tensor = torch.tensor(M_df.values).float()        dtype:torch.Tensor
mutation_features = list(M_df.columns)              dtype:list

========= Beta Read by CSV =======================================

beta_df = pd.read_csv(path, index_col=0)            dtype:DataFrame
beta_tensor = torch.tensor(beta_df.values).float()  dtype:torch.Tensor
signature_names = list(beta_df.index)               dtype:list
mutation_features = list(beta_df.columns)           dtype:list

========= Beta Read by name ======================================

cosmic_df = pd.read_csv(cosmic_path, index_col=0)
beta_df = cosmic_df.loc[beta_name_list]             dtype:DataFrame
beta_tensor = torch.tensor(beta_df.values).float()  dtype:torch.Tensor
signature_names = list(beta_df.index)               dtype:list
mutation_features = list(beta_df.columns)           dtype:list

========= Read RDs files in python ===============================

import pyreadr
result = pyreadr.read_r('/home/azad/Documents/thesis/rds/raw_signa.rds')

======================================================================================================
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

#-----------------------------------------------------------------[PASSESD]
def get_alpha(params):
    alpha = torch.exp(params["alpha"])
    alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))
    return  alpha
    # alpha : torch.Tensor (num_samples X  k)

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
def BIC_zero(params):
    alpha = get_alpha(params)
    theta = torch.sum(params["M"], axis=1)
    beta = params["beta_fixed"]
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
def fixedFilter(alpha_tensor, beta_df, theta_np, fixedLimit):
    # alpha_tensor (inferred alpha) ------------------------- dtype: torch.Tensor
    # beta_df (input beta fixed) ---------------------------- dtype: data.frame
    # theta_np ---------------------------------------------- dtype: numpy

    beta_test_list = list(beta_df.index)
    #k_fixed = len(beta_test_list)

    alpha = np.array(alpha_tensor)
    theta = np.expand_dims(theta_np, axis = 1)
    x = np.multiply(alpha, theta)
    a = np.sum(x, axis=0) / np.sum(x)
    b = [x for x in a if x <= fixedLimit]

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
def denovoFilter(beta_inferred, cosmic_df, denovoLimit):
    # beta_inferred -- dtype: tensor
    # cosmic_path ---- dtype: string
    #cosmic_df = pd.read_csv(cosmic_path, index_col=0)
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
        if maxScore > denovoLimit:
            match.append(cosMatch)
        
    return match    # dtype: list


def stopRun(new_list, old_list, denovo_list):
    # new_list      dtype: list
    # old_list      dtype: list
    # denovo_list   dtype: list
    new_list.sort()
    old_list.sort()
    if new_list==old_list and len(denovo_list)==0:
        return True
    else:
        return False

def betaFixed_perf(B_input, B_fixed_target, B_fixed_inf):
    # B_input ---------- dtype:dataframe
    # B_fixed_target --- dtype:dataframe
    # B_fixed_inf ------ dtype:dataframe

    B_fixed_target_list = list(B_fixed_target.index)
    B_input_list = list(B_input.index)
    B_fixed_inf_list = list(B_fixed_inf.index)
    
    TP_fixed = len(list(set(B_fixed_target_list).intersection(B_fixed_inf_list)))
    FP_fixed = len(list(set(B_fixed_inf_list) - set(B_fixed_target_list)))
    TN_fixed = len(list((set(B_input_list) - set(B_fixed_target_list)) - set(B_fixed_inf_list)))
    FN_fixed = len(list(set(B_fixed_target_list) - set(B_fixed_inf_list)))
    Accuracy = (TP_fixed + TN_fixed)/(TP_fixed + TN_fixed + FP_fixed + FN_fixed)

    return Accuracy # dtype:float


def betaDenovo_perf(inferred, target):
    # inferred ---- dtype:dataframe
    # target ------ dtype:dataframe

    if len(list(target.index)) == len(list(inferred.index)):
        quantity = True
    else:
        quantity = False

    if len(inferred.index) > 0 and len(target.index) > 0:
        scores = {}
        peers = {}
        for infName in list(inferred.index):
            inf = inferred.loc[infName]
            inf_tensor = torch.tensor(inf.values).float()   # dtype: tensor
            inf_tensor = inf_tensor[None, :]                # dtype: tensor (convert from 1D to 2D)
            maxScore = 0
            bestTar = ""
            for tarName in list(target.index):
                tar = target.loc[tarName]                       # pandas Series
                tar_tensor = torch.tensor(tar.values).float()   # dtype: tensor
                tar_tensor = tar_tensor[None, :]                # dtype: tensor (convert from 1D to 2D)

                score = F.cosine_similarity(inf_tensor, tar_tensor).item()
                if score >= maxScore:
                    maxScore = score
                    bestTar = tarName
            peers[infName] = bestTar
            scores[infName] = maxScore
        
        inferred = inferred.rename(index = peers)
        quality = sum(list(scores.values())) / len(list(target.index))
    else:
        quality = -1


    # inferred --- dataframe
    # quantity --- float
    # quality ---- float
    return inferred, quantity, quality




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

#------------------------ DONE! ----------------------------------[passed]
def custom_likelihood_zero(alpha, beta_fixed, M):
    # build full signature profile (beta) matrix
    #beta = torch.cat((beta_fixed, beta_denovo), axis=0)
    likelihood =  dist.Poisson(torch.matmul(torch.matmul(torch.diag(torch.sum(M, axis=1)), alpha), beta_fixed)).log_prob(M)
    #regularization = regularizer(beta_fixed, beta_denovo)
    return likelihood