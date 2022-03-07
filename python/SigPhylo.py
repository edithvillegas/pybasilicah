import numpy as np
import torch
import pyro
import pyro.distributions as dist
import svi
import transfer
import utilities
import torch.nn.functional as F

def single_run(params):

    '''
    params = {
        "M"                 : 0, 
        "beta_fixed"        : 0, 
        "A"                 : 0, 
        "lr"                : 0, 
        "steps_per_iter"    : 0, 
        "max_iter"          : 0, 
        "epsilon"           : 0, 
        "k_denovo"          : 0, 
        "lambda"            : 0
        }
    '''

    M = params["M"]
    num_samples = params["M"].size()[0]
    k_fixed = params["beta_fixed"].size()[0]
    k_denovo = params["k_denovo"]
    landa = params["lambda"]
    max_iter = params["max_iter"]

    
    data = {}   # initialize JSON file (output data)
    LHs = []    # initialize likelihoods list (over iterations)
    BICs = []   # initialize BICs list (over iterations)
    alpha_iters = [] # initialize alphas list (over iterations)

    #======================================================================================
    # step 0 : independent inference ------------------------------------------------------
    #======================================================================================
    print("iteration ", 0)

    #----- variational parameters initialization ----------------------------------------OK
    params["alpha_init"] = dist.Normal(torch.zeros(num_samples, k_denovo + k_fixed), 1).sample()
    params["beta_init"] = dist.Normal(torch.zeros(k_denovo, 96), 1).sample()

    #----- model priors initialization --------------------------------------------------OK
    params["alpha"] = dist.Normal(torch.zeros(num_samples, k_denovo + k_fixed), 1).sample()
    params["beta"] = dist.Normal(torch.zeros(k_denovo, 96), 1).sample()

    svi.inference(params)

    #----- update variational parameters initialization ---------------------------------OK
    params["alpha_init"] = pyro.param("alpha").clone().detach()
    params["beta_init"] = pyro.param("beta").clone().detach()

    #----- update model priors initialization -------------------------------------------OK
    params["alpha"] = pyro.param("alpha").clone().detach()
    params["beta"] = pyro.param("beta").clone().detach()

    #----- get alpha & beta -------------------------------------------------------------OK
    current_alpha, current_beta = utilities.get_alpha_beta(params)

    #----- calculate & save likelihood (list) -------------------------------------------OK
    lh = utilities.log_likelihood(params)
    LHs.append(lh)

    #----- calculate & save BIC (list) --------------------------------------------------OK
    bic = utilities.BIC(params)
    BICs.append(bic)

    #----- save alpha (list) ------------------------------------------------------------OK
    alpha_iters.append(np.asarray(current_alpha))

    #====================================================================================
    # step 1 : inference using transition matrix (iterations)
    #====================================================================================
    for i in range(max_iter):
        print("iteration ", i + 1)

        #----- calculate transfer coeff -------------------------------------------------OK
        transfer_coeff = transfer.calculate_transfer_coeff(params)

        #----- update alpha prior with transfer coeff -----------------------------------OK
        params["alpha"] = torch.matmul(transfer_coeff, params["alpha"])

        svi.inference(params)

        #----- update variational parameters initialization -----------------------------OK
        params["alpha_init"] = pyro.param("alpha").clone().detach()
        params["beta_init"] = pyro.param("beta").clone().detach()

        #----- update model priors initialization ---------------------------------------OK
        params["alpha"] = pyro.param("alpha").clone().detach()
        params["beta"] = pyro.param("beta").clone().detach()

        #----- update alpha & beta ------------------------------------------------------OK
        previous_alpha, previous_beta = current_alpha, current_beta
        current_alpha, current_beta = utilities.get_alpha_beta(params)

        #----- calculate & save likelihood (list) ---------------------------------------OK
        lh = utilities.log_likelihood(params)
        LHs.append(lh)

        #----- calculate & save BIC (list) ---------------------------------------OK
        bic = utilities.BIC(params)
        BICs.append(bic)

        #----- save alpha (list) ------------------------------------------------------------OK
        alpha_iters.append(np.asarray(current_alpha))
        
        #----- convergence test ---------------------------------------------------------
        if (utilities.convergence(current_alpha, previous_alpha, params) == "stop"):
            print("meet convergence criteria, stoped in iteration", i+1)
            break

    #====================================================================================
    # save output data ------------------------------------------------------------------
    #====================================================================================

    #----- phylogeny reconstruction -----------------------------------------------------OK
    M_R = utilities.Reconstruct_M(params)

    #----- save as dictioary ------------------------------------------------------------
    data = {
        "k_denovo": k_denovo, 
        "lambda": landa, 
        "alpha": np.array(current_alpha), 
        "beta": np.array(current_beta), 
        "alphas": np.array(alpha_iters), 
        "log-likes": LHs, 
        "BICs": BICs, 
        "log-like": lh, 
        "BIC": bic, 
        "M_R": np.rint(np.array(M_R)), 
        "cosine": F.cosine_similarity(M, M_R).tolist()
        }

    print("Single Run Finished |", "k_denovo =", k_denovo, "| lambda =", landa)

    return data