import torch
import pyro
import pyro.distributions as dist

from pybasilica import svi
from pybasilica import utilities

#import svi
#import utilities


#------------------------------------------------------------------------------------------------
# run model with single k value
#------------------------------------------------------------------------------------------------
def single_k_run(params):
    '''
    params = {
        "M" :               torch.Tensor
        "beta_fixed" :      torch.Tensor | None
        "k_denovo" :        int
        "lr" :              int
        "steps_per_iter" :  int
    }
    "alpha" :           torch.Tensor    added inside the single_k_run function
    "beta" :            torch.Tensor    added inside the single_k_run function
    "alpha_init" :      torch.Tensor    added inside the single_k_run function
    "beta_init" :       torch.Tensor    added inside the single_k_run function
    '''

    # if No. of inferred signatures and input signatures are zero raise error
    if params["beta_fixed"] is None and params["k_denovo"]==0:
        raise Exception("wrong input!")

    M = params["M"]
    num_samples = params["M"].size()[0]

    if params["beta_fixed"] is None:
        k_fixed=0
    else:
        k_fixed = params["beta_fixed"].size()[0]
    
    k_denovo = int(params["k_denovo"])
    
    #----- variational parameters initialization ----------------------------------------OK
    params["alpha_init"] = dist.Normal(torch.zeros(num_samples, k_denovo + k_fixed), 1).sample()
    if k_denovo > 0:
        params["beta_init"] = dist.Normal(torch.zeros(k_denovo, 96), 1).sample()

    #----- model priors initialization --------------------------------------------------OK
    params["alpha"] = dist.Normal(torch.zeros(num_samples, k_denovo + k_fixed), 1).sample()
    if k_denovo > 0:
        params["beta"] = dist.Normal(torch.zeros(k_denovo, 96), 1).sample()

    svi.inference(params)

    #----- update model priors initialization -------------------------------------------OK
    params["alpha"] = pyro.param("alpha").clone().detach()
    if k_denovo > 0:
        params["beta"] = pyro.param("beta").clone().detach()

    #----- outputs ----------------------------------------------------------------------OK
    alpha_tensor, beta_tensor = utilities.get_alpha_beta(params)  # dtype: torch.Tensor (beta_tensor==0 if k_denovo==0)
    #lh = utilities.log_likelihood(params)           # log-likelihood
    bic = utilities.compute_bic(params)                     # BIC
    #M_R = utilities.Reconstruct_M(params)           # dtype: tensor
    
    return bic, alpha_tensor, beta_tensor


#------------------------------------------------------------------------------------------------
# run model with list of k value
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
        k = int(k)
        if k==0:
            if params["beta_fixed"] is not None:
                params["k_denovo"] = 0
                bic, alpha, beta = single_k_run(params)
                if bic <= BIC_best:
                    BIC_best = bic
                    k_best = k
                    alpha_best = alpha
                    beta_best = beta
            else:
                continue
        else:
            params["k_denovo"] = k
            bic, alpha, beta = single_k_run(params)
            if bic <= BIC_best:
                BIC_best = bic
                k_best = k
                alpha_best = alpha
                beta_best = beta
    return k_best, alpha_best, beta_best

