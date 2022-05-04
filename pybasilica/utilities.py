import numpy as np
import torch
import pyro.distributions as dist
import torch.nn.functional as F

#-----------------------------------------------------------------[PASSESD]
def get_alpha_beta(params):
    alpha = torch.exp(params["alpha"])
    alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))

    if params["k_denovo"] > 0:
        beta = torch.exp(params["beta"])
        beta = beta / (torch.sum(beta, 1).unsqueeze(-1))
    else:
        beta = 0
    return  alpha, beta
    # alpha : torch.Tensor (num_samples X  k)
    # beta  : torch.Tensor (k_denovo    X  96) | ZERO ( 0 )

#-----------------------------------------------------------------[PASSED]
def BIC(params):
    alpha, beta_denovo = get_alpha_beta(params)

    if type(beta_denovo) is int:
    #if beta_denovo==0:
        beta = params["beta_fixed"]
        k_fixed = params["beta_fixed"].shape[0]
    elif type(params["beta_fixed"]) is int:
    #elif params["beta_fixed"]==0:
        beta = beta_denovo
        k_fixed = 0
    else:
        beta = torch.cat((params["beta_fixed"], beta_denovo), axis=0)
        k_fixed = params["beta_fixed"].shape[0]

    theta = torch.sum(params["M"], axis=1)

    log_L_Matrix = dist.Poisson(
        torch.matmul(
            torch.matmul(torch.diag(theta), alpha), 
            beta)
            ).log_prob(params["M"])
    log_L = torch.sum(log_L_Matrix)
    log_L = float("{:.3f}".format(log_L.item()))

    k = (params["M"].shape[0] * (k_fixed + params["k_denovo"])) + (params["k_denovo"] * params["M"].shape[1])
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
    
    if type(beta_inferred) is int:
        return match

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


#------------------------ DONE! ----------------------------------[passed]
# note: just check the order of kl-divergence arguments and why the value is negative
def regularizer(beta_fixed, beta_denovo):
    loss = 0
    for fixed in beta_fixed:
        for denovo in beta_denovo:
            loss += F.kl_div(fixed, denovo, reduction="batchmean").item()

    return loss

#------------------------ DONE! ----------------------------------[passed]
def custom_likelihood(M, alpha, beta_fixed, beta_denovo):
    # build full signature profile (beta) matrix

    if type(beta_fixed) is int:
    #if beta_fixed==0:
        beta = beta_denovo
        regularization = 0

    elif type(beta_denovo) is int:
    #elif beta_denovo==0:
        beta = beta_fixed
        regularization = 0

    else:
        beta = torch.cat((beta_fixed, beta_denovo), axis=0)
        regularization = regularizer(beta_fixed, beta_denovo)
    
    likelihood =  dist.Poisson(torch.matmul(torch.matmul(torch.diag(torch.sum(M, axis=1)), alpha), beta)).log_prob(M)
    
    return likelihood + regularization
