from pybasilica.svi import PyBasilica
#from svi import PyBasilica


def single_run(x, k_denovo, lr=0.05, n_steps=500, groups=None, beta_fixed=None):
    
    obj = PyBasilica(x, k_denovo, lr, n_steps, groups=groups, beta_fixed=beta_fixed)
    obj._fit()
    minBic = obj.bic
    bestRun = obj
    for i in range(2):
        obj = PyBasilica(x, k_denovo, lr, n_steps, groups=groups, beta_fixed=beta_fixed)
        obj._fit()

        if obj.bic < minBic:
            minBic = obj.bic
            bestRun = obj

    return bestRun


def fit(x, k_list=[0,1,2,3,4,5], lr=0.05, n_steps=500, groups=None, beta_fixed=None):

    if isinstance(k_list, list):
        if len(k_list) > 0:
            pass
        else:
            raise Exception("k_list is an empty list!")
    elif isinstance(k_list, int):
        k_list = [k_list]
    else:
        raise Exception("invalid k_list datatype")
    
    #len(k_list)==1

    '''
    if isinstance(k_list, list) and len(k_list)>0:
        pass
    elif not isinstance(k_list, list):
        pass
    else:
        raise Exception("invalid k_list argument")
    '''

    #minBic = 10000000
    #bestRun = None

    obj = single_run(x=x, k_denovo=k_list[0], lr=lr, n_steps=n_steps, groups=groups, beta_fixed=beta_fixed)
    minBic = obj.bic
    bestRun = obj

    for k in k_list[1:]:
        try:
            obj = single_run(x=x, k_denovo=k, lr=lr, n_steps=n_steps, groups=groups, beta_fixed=beta_fixed)

            if obj.bic < minBic:
                minBic = obj.bic
                bestRun = obj
        except:
            continue

    try:
        bestRun._convert_to_dataframe(x, beta_fixed)
    except:
        raise Exception("No run, please take care of inputs, probably k_list!")

    return bestRun



#def stop()



'''
#import utilities

import torch
import pyro
import pyro.distributions as dist

from pybasilica import svi
from pybasilica import utilities



#------------------------------------------------------------------------------------------------
# run model with single k value
#------------------------------------------------------------------------------------------------
def single_k_run(params):
    #params = {
    #    "M" :               torch.Tensor
    #    "beta_fixed" :      torch.Tensor | None
    #    "k_denovo" :        int
    #    "lr" :              int
    #    "steps_per_iter" :  int
    #}
    #"alpha" :           torch.Tensor    added inside the single_k_run function
    #"beta" :            torch.Tensor    added inside the single_k_run function
    #"alpha_init" :      torch.Tensor    added inside the single_k_run function
    #"beta_init" :       torch.Tensor    added inside the single_k_run function

    # if No. of inferred signatures and input signatures are zero raise error
    #if params["beta_fixed"] is None and params["k_denovo"]==0:
    #    raise Exception("Error: both denovo and fixed signatures are zero")


    #-----------------------------------------------------
    #M = params["M"]
    num_samples = params["M"].size()[0]

    if params["beta_fixed"] is None:
        k_fixed = 0
    else:
        k_fixed = params["beta_fixed"].size()[0]
    
    k_denovo = params["k_denovo"]

    if k_fixed + k_denovo == 0:
        raise Exception("Error: both denovo and fixed signatures are zero")
    #-----------------------------------------------------

    
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
    
    #params = {
    #    "M" :               torch.Tensor
    #    "beta_fixed" :      torch.Tensor
    #    "lr" :              int
    #    "steps_per_iter" :  int
    #}
    #"k_denovo" : int    added inside the multi_k_run function
    

    bic_best = 10000000000
    k_best = -1

    for k_denovo in k_list:
        try:
            params["k_denovo"] = int(k_denovo)
            bic, alpha, beta = single_k_run(params)
            if bic <= bic_best:
                bic_best = bic
                k_best = k_denovo
                alpha_best = alpha
                beta_best = beta

        except Exception:
            continue
    
    return k_best, alpha_best, beta_best

'''

