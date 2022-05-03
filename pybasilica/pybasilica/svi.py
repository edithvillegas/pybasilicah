import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.distributions as dist
import torch
from pybasilica import utilities


#------------------------------------------------------------------------------------------------
# Model [PASSED]
#------------------------------------------------------------------------------------------------

def model(params):
    '''
    params = {
        "M" :           torch.Tensor
        "beta_fixed" :  torch.Tensor
        "k_denovo" :    int
        "alpha" :       torch.Tensor
        "beta" :        torch.Tensor
    }
    '''

    num_samples = params["M"].size()[0]

    if type(params["beta_fixed"]) is int:
    #if params["beta_fixed"]==0:
        beta_fixed = 0
        k_fixed = 0
    else:
        beta_fixed = params["beta_fixed"]
        k_fixed = beta_fixed.size()[0]
    
    k_denovo = params["k_denovo"]
    
    # alpha is relative exposure (normalized or percentages of signature activity)
    # theta encodes the total number of mutations in each branch
    # parametarize the activity matrix as : theta * alpha

    # sample from the alpha prior
    with pyro.plate("K", k_fixed + k_denovo):   # columns
        with pyro.plate("N", num_samples):      # rows
            alpha = pyro.sample("activities", dist.Normal(params["alpha"], 1))
    
    alpha = torch.exp(alpha)                                # enforce non negativity
    alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))     # normalize
    

    #-------------------------- beta denovo is NULL ----------------------------------------
    if k_denovo==0:
        beta_denovo=0
    
    #-------------------------- beta denovo is present -------------------------
    else:
        # sample from the beta prior
        with pyro.plate("contexts", 96):            # columns
            with pyro.plate("k_denovo", k_denovo):  # rows
                beta_denovo = pyro.sample("denovo_signatures", dist.Normal(params["beta"], 1))

        beta_denovo = torch.exp(beta_denovo)                                    # enforce non negativity
        beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))   # normalize


    # compute the custom likelihood
    with pyro.plate("context", 96):
        with pyro.plate("sample", num_samples):
            pyro.factor("obs", utilities.custom_likelihood(params["M"], alpha, beta_fixed, beta_denovo))




def guide(params):
    '''
    params = {
        "M" :           torch.Tensor
        "beta_fixed" :  torch.Tensor
        "k_denovo" :    int
        "alpha_init" :  torch.Tensor
        "beta_init" :   torch.Tensor
    }
    '''
    num_samples = params["M"].size()[0]
    k_denovo = params["k_denovo"]

    if type(params["beta_fixed"]) is int:
    #if params["beta_fixed"]==0:
        k_fixed = 0
    else:
        k_fixed = params["beta_fixed"].size()[0]


    with pyro.plate("K", k_fixed + k_denovo):
        with pyro.plate("N", num_samples):
            alpha = pyro.param("alpha", params["alpha_init"])
            pyro.sample("activities", dist.Delta(alpha))

    if k_denovo!=0:
        with pyro.plate("contexts", 96):
            with pyro.plate("k_denovo", k_denovo):
                beta = pyro.param("beta", params["beta_init"])
                pyro.sample("denovo_signatures", dist.Delta(beta))



#------------------------------------------------------------------------------------------------
# inference [PASSED]
#------------------------------------------------------------------------------------------------
'''
params = {
    "M" :               torch.Tensor
    "beta_fixed" :      torch.Tensor
    "k_denovo" :        int
    "alpha" :           torch.Tensor
    "beta" :            torch.Tensor
    "alpha_init" :      torch.Tensor
    "beta_init" :       torch.Tensor
    "lr" :              int
    "steps_per_iter" :  int
}
'''

def inference(params):
    
    pyro.clear_param_store()  # always clear the store before the inference

    # learning global parameters

    adam_params = {"lr": params["lr"]}
    optimizer = Adam(adam_params)
    elbo = Trace_ELBO()

    #if params["k_denovo"] > 0:
    #    svi = SVI(model, guide, optimizer, loss=elbo)
    #else:
    #    svi = SVI(model_zero, guide_zero, optimizer, loss=elbo)

    svi = SVI(model, guide, optimizer, loss=elbo)

#   inference - do gradient steps
    for step in range(params["steps_per_iter"]):
        loss = svi.step(params)




'''

#---------------------------------------------------------------------------------

# NOT USED ANYMORE
def model(params):
    
    num_samples = params["M"].size()[0]
    beta_fixed = params["beta_fixed"]
    k_fixed = beta_fixed.size()[0]
    k_denovo = params["k_denovo"]

    # parametarize the activity matrix as : theta * alpha
    # theta encodes the total number of mutations in each branch
    # alpha is relative exposure (normalized or percentages of signature activity)

    # sample from the alpha prior
    with pyro.plate("K", k_fixed + k_denovo):   # columns
        with pyro.plate("N", num_samples):      # rows
            alpha = pyro.sample("activities", dist.Normal(params["alpha"], 1))

    # sample from the beta prior
    with pyro.plate("contexts", 96):            # columns
        with pyro.plate("k_denovo", k_denovo):  # rows
            beta_denovo = pyro.sample("denovo_signatures", dist.Normal(params["beta"], 1))

    # enforce non negativity
    alpha = torch.exp(alpha)
    beta_denovo = torch.exp(beta_denovo)

    # normalize
    alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))
    beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))

    # compute the custom likelihood
    with pyro.plate("context", 96):
        with pyro.plate("sample", num_samples):
            pyro.factor("obs", utilities.custom_likelihood(alpha, beta_fixed, beta_denovo, params["M"]))


#------------------------------------------------------------------------------------------------
# Guide [PASSED]
#------------------------------------------------------------------------------------------------
===============================================
params = {
    "M" :           torch.Tensor
    "beta_fixed" :  torch.Tensor
    "k_denovo" :    int
    "alpha_init" :  torch.Tensor
    "beta_init" :   torch.Tensor
}
===============================================

# NOT USED ANYMORE
def guide(params):

    num_samples = params["M"].size()[0]
    k_fixed = params["beta_fixed"].size()[0]
    k_denovo = params["k_denovo"]

    with pyro.plate("K", k_fixed + k_denovo):
        with pyro.plate("N", num_samples):
            alpha = pyro.param("alpha", params["alpha_init"])
            pyro.sample("activities", dist.Delta(alpha))

    with pyro.plate("contexts", 96):
        with pyro.plate("k_denovo", k_denovo):
            beta = pyro.param("beta", params["beta_init"])
            pyro.sample("denovo_signatures", dist.Delta(beta))


#------------------------------------------------------------------------------------------------
# Model for k_denovo = 0 [PASSED]
#------------------------------------------------------------------------------------------------
===============================================
params = {
    "M" :           torch.Tensor
    "beta_fixed" :  torch.Tensor
    "k_denovo" :    int ------------> zero
    "alpha" :       torch.Tensor
    "beta" :        torch.Tensor ---> eliminate
}
===============================================

# NOT USED ANYMORE
def model_zero(params):
    
    num_samples = params["M"].size()[0]
    beta_fixed = params["beta_fixed"]
    k_fixed = beta_fixed.size()[0]
    #k_denovo = params["k_denovo"]

    # parametarize the activity matrix as : theta * alpha
    # theta encodes the total number of mutations in each branch
    # alpha is relative exposure (normalized or percentages of signature activity)

    # sample from the alpha prior
    with pyro.plate("K", k_fixed):   # columns
        with pyro.plate("N", num_samples):      # rows
            alpha = pyro.sample("activities", dist.Normal(params["alpha"], 1))

    alpha = torch.exp(alpha)                            # enforce non negativity
    alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1)) # normalize

    # compute the custom likelihood
    with pyro.plate("context", 96):
        with pyro.plate("sample", num_samples):
            pyro.factor("obs", utilities.custom_likelihood_zero(alpha, beta_fixed, params["M"]))


#------------------------------------------------------------------------------------------------
# Guide for k_denovo = 0 [PASSED]
#------------------------------------------------------------------------------------------------
===============================================
params = {
    "M" :           torch.Tensor
    "beta_fixed" :  torch.Tensor
    "k_denovo" :    int ------------> zero
    "alpha_init" :  torch.Tensor
    "beta_init" :   torch.Tensor ---> eliminate
}
===============================================

# NOT USED ANYMORE
def guide_zero(params):

    num_samples = params["M"].size()[0]
    k_fixed = params["beta_fixed"].size()[0]
    #k_denovo = params["k_denovo"]

    with pyro.plate("K", k_fixed):
        with pyro.plate("N", num_samples):
            alpha = pyro.param("alpha", params["alpha_init"])
            pyro.sample("activities", dist.Delta(alpha))

'''