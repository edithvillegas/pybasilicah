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
from statistics import mean

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


#-----------------------------------------------------------------[QC:PASSED]
# divide cosmic signatures into two parts:
# 1st part: considered as cosmic signatures
# 2nd part: considered as denovo signatures

def cosmic_denovo(cosmic_path):
    # cosmic_path --- dtype: str
    full_cosmic_df = pd.read_csv(cosmic_path, index_col=0)
    full_cosmic_list = list(full_cosmic_df.index)

    cosmic_list = random.sample(full_cosmic_list, k=50)
    denovo_list = list(set(full_cosmic_list) - set(cosmic_list))

    cosmic_df = full_cosmic_df.loc[cosmic_list] # cosmic signatures (dtype: dataframe)
    denovo_df = full_cosmic_df.loc[denovo_list] # denovo signatures (dtype: dataframe)

    return cosmic_df, denovo_df
    # cosmic_df --- dtype: dataframe)
    # denovo_df --- dtype: dataframe)

#-----------------------------------------------------------------[QC:PASSED]
def target_generator(profile, cosmic_df, denovo_df):
    # profile ------- {"A", "B", "C"}
    # cosmic_df ----- dtype: dataframe
    # denovo_df ----- dtype: dataframe

    num_samples = random.randint(3, 7)

    if profile=="A":
        fixed_num = random.randint(3, 5)
        denovo_num = random.randint(0, 2)
    elif profile=="B":
        fixed_num = random.randint(0, 2)
        denovo_num = random.randint(3, 5)
    elif profile=="C":
        fixed_num = random.randint(1, 4)
        denovo_num = random.randint(1, 4)
    else:
        print("NOT VALID!")
        return -1

    cosmic_list = list(cosmic_df.index)
    denovo_list = list(denovo_df.index)
    mutation_features = list(cosmic_df.columns)

    # beta fixed ------------------------------------------------------
    if fixed_num > 0:
        target_fixed_list = random.sample(cosmic_list, k=fixed_num)
        beta_fixed_df = cosmic_df.loc[target_fixed_list]
    else:
        beta_fixed_df = pd.DataFrame(columns=mutation_features)
    
    # beta denovo -----------------------------------------------------
    if denovo_num > 0:
        target_denovo_list = random.sample(denovo_list, k=denovo_num)
        beta_denovo_df = denovo_df.loc[target_denovo_list]

        denovo_labels = []
        for i in range(len(target_denovo_list)):
            denovo_labels.append(target_denovo_list[i] + '_D')
        beta_denovo_df.index = denovo_labels
    else:
        beta_denovo_df = pd.DataFrame(columns=mutation_features)


    if beta_denovo_df.empty:
        beta_df = beta_fixed_df
    elif beta_fixed_df.empty:
        beta_df = beta_denovo_df
    else:
        beta_df = pd.concat([beta_fixed_df, beta_denovo_df], axis=0)

    signatures = list(beta_df.index)
    beta_tensor = torch.tensor(beta_df.values).float()

    #------- alpha ----------------------------------------------
    matrix = np.random.rand(num_samples, len(signatures))
    alpha_tensor = torch.tensor(matrix / matrix.sum(axis=1)[:, None]).float()
    alpha_np = np.array(alpha_tensor)
    alpha_df = pd.DataFrame(alpha_np, columns=signatures)

    #------- theta ----------------------------------------------
    theta = random.sample(range(1000, 4000), k=num_samples) # dtype:list
    
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

    # phylogeny
    M_np = np.array(M_tensor)
    M_df = pd.DataFrame(M_np, columns=mutation_features)
    M_df = M_df.astype(int)

    # return all in dataframe format
    return M_df, alpha_df, beta_fixed_df, beta_denovo_df
    # M_df ------------- dtype: dataframe
    # alpha_df --------- dtype: dataframe
    # beta_fixed_df ---- dtype: dataframe, could be empty (if empty --> beta_denovo_df != empty)
    # beta_denovo_df --- dtype: dataframe, could be empty (if empty --> beta_fixed_df != empty)

#-----------------------------------------------------------------[QC:PASSED]
def beta_input_generator(profile, beta_fixed_df, beta_denovo_df, cosmic_df):
    # profile ---------- {"X", "Y", "Z"}
    # beta_fixed_df ---- dtype: dataframe
    # beta_denovo_df --- dtype: dataframe
    # cosmic_df -------- dtype: dataframe

    # TARGET DATA -----------------------------------------------------------------------------
    beta_fixed_names = list(beta_fixed_df.index)    # target beta signatures names (dtype: list)
    k_fixed_target = len(beta_fixed_names)
    beta_denovo_names = list(beta_denovo_df.index)  # target beta signatures names (dtype: list)
    k_denovo_target = len(beta_denovo_names)

    # common fixed signatures (target intersect test)
    # different fixed signatures (test minus target)
    if profile=="X":
        if k_fixed_target > 0:
            k_overlap = random.randint(1, k_fixed_target)
            k_extra = 0
        else:
            k_overlap = 0
            k_extra = random.randint(1, k_denovo_target)

    elif profile=="Y":
        if k_fixed_target > 0:
            k_overlap = random.randint(1, k_fixed_target)
            k_extra = random.randint(1, k_fixed_target)
        else:
            k_overlap = 0
            k_extra = random.randint(1, k_denovo_target)

    elif profile=="Z":
        k_overlap = 0
        k_extra = random.randint(1, k_fixed_target + k_denovo_target)
    else:
        print("NOT VALID!")
        return "NA"
    
    cosmic_names = list(cosmic_df.index)    # cosmic signatures names (dtype: list)

    # exclude beta target fixed signatures from cosmic
    for signature in beta_fixed_names:
        cosmic_names.remove(signature)
    
    # common fixed signatures list
    if k_overlap > 0:
        overlap_sigs = random.sample(beta_fixed_names, k=k_overlap)
    else:
        overlap_sigs = []
    
    # different fixed signatures list
    if k_extra > 0:
        extra_sigs = random.sample(cosmic_names, k=k_extra)
    else:
        extra_sigs = []
    
    beta_input = cosmic_df.loc[overlap_sigs + extra_sigs]

    return beta_input   # dtype: dataframe


#-----------------------------------------------------------------[PASSED]
def input_generator(Tprofile, Iprofile, cosmic_path_org):

    cosmic_df, denovo_df = cosmic_denovo(cosmic_path_org)
    M_df, alpha_df, beta_fixed_df, beta_denovo_df = target_generator(Tprofile, cosmic_df, denovo_df)
    beta_input_df = beta_input_generator(Iprofile, beta_fixed_df, beta_denovo_df, cosmic_df)
    #beta_df = pd.concat([beta_fixed_df, beta_denovo_df], axis=0)

    data = {
        "M" : M_df,                         # dataframe
        "alpha" : alpha_df,                 # dataframe
        "beta_fixed" : beta_fixed_df,       # dataframe
        "beta_denovo" : beta_denovo_df,     # dataframe
        "beta_input" : beta_input_df,       # dataframe
        "cosmic_df" : cosmic_df             # dataframe
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
def denovoFilter(beta_inferred, cosmic_df):
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
        if maxScore > 0.8:
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


def BaySiLiCo(M, B_input, k_list, cosmic_df):
    # M ------------- dataframe
    # B_fixed ------- dataframe
    # k_list -------- list
    # cosmic_path --- str
    #cosmic_df = pd.read_csv(cosmic_path, index_col=0)
    theta = np.sum(M.values, axis=1)
    params = {
        "M" :               torch.tensor(M.values).float(), 
        "beta_fixed" :      torch.tensor(B_input.values).float(), 
        "lr" :              0.05, 
        "steps_per_iter" :  500
        }

    counter = 1
    while True:
        print("Loop", counter)

        # k_list --- dtype: list
        k_inf, A_inf, B_inf = run.multi_k_run(params, k_list)
        # k_inf --- dtype: int
        # A_inf --- dtype: torch.Tensor
        # B_inf --- dtype: torch.Tensor

        # A_inf ----- dtype: torch.Tensor
        # B_fixed --- dtype: data.frame
        # theta ----- dtype: numpy
        B_input_sub = fixedFilter(A_inf, B_input, theta)
        # output ---- dtype: list
        
        if k_inf > 0:
            # B_inf ---------- dtype: torch.Tensor
            # cosmic_path ---- dtype: string
            B_input_new = denovoFilter(B_inf, cosmic_df)
            # B_fixed_new --- dtype: list
        else:
            B_input_new = []


        # B_input_sub ----------- dtype: list
        # list(B_input.index) --- dtype: list
        # B_input_new ----------- dtype: list
        if stopRun(B_input_sub, list(B_input.index), B_input_new):
            signatures_inf = []
            for k in range(k_inf):
                signatures_inf.append("Unknown"+str(k+1))
            signatures = list(B_input.index) + signatures_inf
            mutation_features = list(B_input.columns)

            # alpha
            A_inf_np = np.array(A_inf)
            A_inf_df = pd.DataFrame(A_inf_np, columns=signatures)   # dataframe

            # beta
            if B_inf=="NA":
                B_inf_denovo_df = pd.DataFrame(columns=mutation_features)
            else:
                B_inf_denovo_np = np.array(B_inf)
                B_inf_denovo_df = pd.DataFrame(B_inf_denovo_np, index=signatures_inf, columns=mutation_features)
            #B_full = torch.cat((params["beta_fixed"], B_inf), axis=0)
            
            B_inf_fixed_df = B_input    # dataframe

            return A_inf_df, B_inf_fixed_df, B_inf_denovo_df
            # A_inf_df ---------- dtype: dataframe
            # B_inf_fixed_df ---- dtype: dataframe
            # B_inf_denovo_df --- dtype: dataframe

        if (B_input_sub + B_input_new) == []:
            print("B_fixed_sub + B_fixed_new = empty")
        else:
            B_input = cosmic_df.loc[B_input_sub + B_input_new]  # dtype: dataframe
            params["beta_fixed"] = torch.tensor(B_input.values).float()
        
        counter += 1


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

    if len(inferred.index) > 0:
        scores = {}
        peers = {}
        for infName in list(inferred.index):
            inf = inferred.loc[infName]
            inf_tensor = torch.tensor(inf.values).float()   # dtype: tensor
            inf_tensor = inf_tensor[None, :]                # dtype: tensor (convert from 1D to 2D)
            maxScore = 0
            bestTar = ""
            print("===========================")
            print("denovo:", infName)
            for tarName in list(target.index):
                tar = target.loc[tarName]              # pandas Series
                tar_tensor = torch.tensor(tar.values).float()   # dtype: tensor
                tar_tensor = tar_tensor[None, :]                # dtype: tensor (convert from 1D to 2D)

                score = F.cosine_similarity(inf_tensor, tar_tensor).item()
                if score >= maxScore:
                    maxScore = score
                    bestTar = tarName
                    print("target:", tarName, "| Score:", score)
            peers[infName] = bestTar
            scores[infName] = maxScore
        
        inferred = inferred.rename(index = peers)
        print("scores:", list(scores.values()))
        quality = mean(list(scores.values()))
    else:
        quality = -1


    # inferred --- dataframe
    # quantity --- float
    # quality ---- float
    return inferred, quantity, quality


def run_simulated(Tprofile, Iprofile, cos_path_org):

    # ========== INPUT ==========================================================
    input_data = input_generator(Tprofile, Iprofile, cos_path_org)
    M = input_data["M"]                     # dataframe
    A = input_data["alpha"]                 # dataframe
    B_fixed = input_data["beta_fixed"]      # dataframe
    B_denovo = input_data["beta_denovo"]    # dataframe
    B_input = input_data["beta_input"]      # dataframe
    cosmic_df = input_data["cosmic_df"]     # dataframe
    k_list = [0, 1, 2, 3, 4, 5]             # list

    # ========== OUTPUT =========================================================
    A_inf, B_fixed_inf, B_denovo_inf = BaySiLiCo(M, B_input, k_list, cosmic_df) # all dataframe

    # ========== Metrics ========================================================
    B_fixed_accuracy = betaFixed_perf(B_input, B_fixed, B_fixed_inf)
    B_denovo_inf_labeled, B_denovo_quantity, B_denovo_quality = betaDenovo_perf(B_denovo, B_denovo_inf)

    theta_tensor = torch.sum(torch.Tensor(M.values).float(), axis=1)
    A_tensor = torch.Tensor(A_inf.values).float()
    beta_df = pd.concat([B_fixed_inf, B_denovo_inf], axis=0)
    B_tensor = torch.tensor(beta_df.values).float()
    M_r = torch.matmul(torch.matmul(torch.diag(theta_tensor), A_tensor), B_tensor)
    gof = mean(F.cosine_similarity(torch.tensor(M.values).float(), M_r).tolist())

    output = {
        "A_target"          : A,                    # dataframe
        "B_fixed_target"    : B_fixed,              # dataframe
        "B_denovo_target"   : B_denovo,             # dataframe
        "B_input"           : B_input,              # dataframe
        "A_inf"             : A_inf,                # dataframe
        "B_fixed_inf"       : B_fixed_inf,          # dataframe
        "B_denovo_inf"      : B_denovo_inf_labeled, # dataframe
        "GoodnessofFit"     : gof,                      # float
        "Accuracy"          : B_fixed_accuracy,         # float
        "Quantity"          : B_denovo_quantity,        # bool
        "Quality"           : B_denovo_quality,         # float
        }

    return output


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