import torch
import pyro
import pyro.distributions as dist
import svi
import utilities


#------------------------------------------------------------------------------------------------
# single k run [PASSED]
#------------------------------------------------------------------------------------------------
def single_k_run(params):
    '''
    params = {
        "M" :               torch.Tensor
        "beta_fixed" :      torch.Tensor
        "k_denovo" :        int
        "lr" :              int
        "steps_per_iter" :  int
    }
    "alpha" :           torch.Tensor    added inside the single_k_run function
    "beta" :            torch.Tensor    added inside the single_k_run function
    "alpha_init" :      torch.Tensor    added inside the single_k_run function
    "beta_init" :       torch.Tensor    added inside the single_k_run function
    '''

    M = params["M"]
    num_samples = params["M"].size()[0]
    k_fixed = params["beta_fixed"].size()[0]
    k_denovo = params["k_denovo"]
    
    #----- variational parameters initialization ----------------------------------------OK
    params["alpha_init"] = dist.Normal(torch.zeros(num_samples, k_denovo + k_fixed), 1).sample()
    params["beta_init"] = dist.Normal(torch.zeros(k_denovo, 96), 1).sample()

    #----- model priors initialization --------------------------------------------------OK
    params["alpha"] = dist.Normal(torch.zeros(num_samples, k_denovo + k_fixed), 1).sample()
    params["beta"] = dist.Normal(torch.zeros(k_denovo, 96), 1).sample()

    svi.inference(params)

    #----- update model priors initialization -------------------------------------------OK
    params["alpha"] = pyro.param("alpha").clone().detach()
    params["beta"] = pyro.param("beta").clone().detach()

    #----- outputs ----------------------------------------------------------------------OK
    alpha_tensor, beta_tensor = utilities.get_alpha_beta(params)  # dtype: torch.Tensor
    #lh = utilities.log_likelihood(params)           # log-likelihood
    bic = utilities.BIC(params)                     # BIC
    #M_R = utilities.Reconstruct_M(params)           # dtype: tensor
    
    return bic, alpha_tensor, beta_tensor


#------------------------------------------------------------------------------------------------
# single k run for k_denovo = 0 [PASSED]
#------------------------------------------------------------------------------------------------
def single_k_run_zero(params):
    '''
    params = {
        "M" :               torch.Tensor
        "beta_fixed" :      torch.Tensor
        "k_denovo" :        int ----------> 0
        "lr" :              int
        "steps_per_iter" :  int
    }
    "alpha" :           torch.Tensor    added inside the single_k_run function
    "beta" :            torch.Tensor    added inside the single_k_run function ----> eliminte
    "alpha_init" :      torch.Tensor    added inside the single_k_run function
    "beta_init" :       torch.Tensor    added inside the single_k_run function ----> eliminte
    '''

    M = params["M"]
    num_samples = params["M"].size()[0]
    k_fixed = params["beta_fixed"].size()[0]
    k_denovo = params["k_denovo"]
    
    #----- variational parameters initialization ----------------------------------------OK
    params["alpha_init"] = dist.Normal(torch.zeros(num_samples, k_denovo + k_fixed), 1).sample()
    #params["beta_init"] = dist.Normal(torch.zeros(k_denovo, 96), 1).sample()

    #----- model priors initialization --------------------------------------------------OK
    params["alpha"] = dist.Normal(torch.zeros(num_samples, k_denovo + k_fixed), 1).sample()
    #params["beta"] = dist.Normal(torch.zeros(k_denovo, 96), 1).sample()

    svi.inference(params)

    #----- update model priors initialization -------------------------------------------OK
    params["alpha"] = pyro.param("alpha").clone().detach()
    #params["beta"] = pyro.param("beta").clone().detach()

    #----- outputs ----------------------------------------------------------------------OK
    alpha_tensor = utilities.get_alpha(params)  # dtype: torch.Tensor
    #lh = utilities.log_likelihood(params)           # log-likelihood
    bic = utilities.BIC_zero(params)            # BIC
    #M_R = utilities.Reconstruct_M(params)           # dtype: tensor
    
    return bic, alpha_tensor

#------------------------------------------------------------------------------------------------
# multi k run for k_denovo = 0 and higher [PASSED]
#------------------------------------------------------------------------------------------------
def multi_k_run(params, k_list):
    '''
    params = {
        "M" :               torch.Tensor
        "beta_fixed" :      torch.Tensor
        "lr" :              int
        "steps_per_iter" :  int
    }
    "k_denovo" : int    added inside the multi_k_run function
    '''

    BIC_best = 10000000000
    k_best = -1

    for k in k_list:
        if k==0:
            params["k_denovo"] = 0
            bic, alpha = single_k_run_zero(params)
            #print("bic(k = 0)", bic)
            if bic <= BIC_best:
                BIC_best = bic
                k_best = k
                alpha_best = alpha
                beta_best = "NA"
        else:
            params["k_denovo"] = k
            bic, alpha, beta = single_k_run(params)
            #print("bic(k =", k, ")", bic)
            if bic <= BIC_best:
                BIC_best = bic
                k_best = k
                alpha_best = alpha
                beta_best = beta
    return k_best, alpha_best, beta_best
