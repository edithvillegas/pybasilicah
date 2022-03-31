import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.distributions as dist
import torch
import utilities


#------------------------------------------------------------------------------------------------
# model
#------------------------------------------------------------------------------------------------

def model(params):
    
    num_samples = params["M"].size()[0]
    beta_fixed = params["beta_fixed"]
    k_fixed = beta_fixed.size()[0]
    k_denovo = params["k_denovo"]
    theta = torch.sum(params["M"], axis=1)

    # parametrize the activity matrix as theta*alpha
    # theta encodes the total number of mutations of the branches
    # alpha is relative exposure (percentages of signature activity)

    # sample from the alpha prior
    with pyro.plate("K", k_denovo + k_fixed):   # columns
        with pyro.plate("N", num_samples):      # rows
            alpha = pyro.sample("activities", dist.Normal(params["alpha"], 1))

    # sample from the beta prior
    with pyro.plate("contexts", 96):            # columns
        with pyro.plate("K_denovo", k_denovo):  # rows
            beta_denovo = pyro.sample("extra_signatures", dist.Normal(params["beta"], 1))

    # enforce non negativity
    alpha = torch.exp(alpha)
    beta_denovo = torch.exp(beta_denovo)

    # normalize
    alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))
    beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))

    # build full signature profile (beta) matrix
    #beta = torch.cat((beta_fixed, beta_denovo), axis=0)

    # compute the custom likelihood
    with pyro.plate("context", 96):
        with pyro.plate("sample", num_samples):
            pyro.factor("obs", utilities.custom_likelihood(alpha, beta_denovo, beta_fixed, params["M"]))

#------------------------------------------------------------------------------------------------
# guide
#------------------------------------------------------------------------------------------------

def guide(params):

    num_samples = params["M"].size()[0]
    k_fixed = params["beta_fixed"].size()[0]
    k_denovo = params["k_denovo"]

    with pyro.plate("K", k_denovo + k_fixed):
        with pyro.plate("N", num_samples):
            alpha = pyro.param("alpha", params["alpha_init"])
            pyro.sample("activities", dist.Delta(alpha))

    with pyro.plate("contexts", 96):
        with pyro.plate("K_denovo", k_denovo):
            beta = pyro.param("beta", params["beta_init"])
            pyro.sample("extra_signatures", dist.Delta(beta))


#------------------------------------------------------------------------------------------------
# inference
#------------------------------------------------------------------------------------------------

def inference(params):
    
    pyro.clear_param_store()  # always clear the store before the inference

    # learning global parameters

    adam_params = {"lr": params["lr"]}
    optimizer = Adam(adam_params)
    elbo = Trace_ELBO()

    svi = SVI(model, guide, optimizer, loss=elbo)

#   inference - do gradient steps
    for step in range(params["steps_per_iter"]):
        loss = svi.step(params)

