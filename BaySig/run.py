import numpy as np
import torch
import pyro
import pyro.distributions as dist
import svi
import utilities
import torch.nn.functional as F

'''
Argument Template
params = {
    "M"                 : 0, 
    "beta_fixed"        : 0, 
    "lr"                : 0, 
    "steps_per_iter"    : 0, 
    "k_denovo"          : 0, 
    "cosmic_path"       : 0
    }
'''

def single_run(params):

    M = params["M"]
    num_samples = params["M"].size()[0]
    k_fixed = params["beta_fixed"].size()[0]
    k_denovo = params["k_denovo"]

    
    data = {}   # initialize JSON file (output data)

    #print("| k_denovo =", k_denovo, "| Start Running")

    #======================================================================================
    # step 0 : independent inference ------------------------------------------------------
    #======================================================================================
    #print("iteration ", 0)
    
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

    #----- get alpha & beta -------------------------------------------------------------OK
    alpha, beta = utilities.get_alpha_beta(params)

    #----- calculate & save likelihood (list) -------------------------------------------OK
    lh = utilities.log_likelihood(params)

    #----- calculate & save BIC (list) --------------------------------------------------OK
    bic = utilities.BIC(params)

    #====================================================================================
    # save output data ------------------------------------------------------------------
    #====================================================================================

    #----- phylogeny reconstruction -----------------------------------------------------OK
    M_R = utilities.Reconstruct_M(params)

    #----- save as dictioary ------------------------------------------------------------
    data = {
        "k_denovo": k_denovo, 
        "alpha": np.array(alpha), 
        "beta": np.array(beta), 
        "log-like": lh, 
        "BIC": bic, 
        "M_R": np.rint(np.array(M_R)), 
        "cosine": F.cosine_similarity(M, M_R).tolist()
        }

    #return data
    return bic, alpha, beta