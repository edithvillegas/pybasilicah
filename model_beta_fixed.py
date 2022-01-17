import pandas as pd
import torch
import pyro
import pyro.distributions as dist
import utilities

def model(params):
    
    num_samples = params["M"].size()[0]
    beta = params["beta_fixed"]
    K = beta.size()[0]
    theta = torch.sum(params["M"], axis=1)

    # parametrize the activity matrix as theta*alpha, where
    # theta encodes the total number of mutations of the branches
    # alpha's are percentages of signature activity

    # sample alpha from a normal distribution using alpha prior
    with pyro.plate("K", K):
        with pyro.plate("N", num_samples):
            alpha = pyro.sample("activities", dist.Normal(params["alpha"], 1))

    # enforce non negativity
    alpha = torch.exp(alpha)

    # normalize
    alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))

    # write the likelihood
    with pyro.plate("context", 96):
        with pyro.plate("sample", num_samples):
            pyro.sample("obs", 
                        dist.Poisson(torch.matmul(torch.matmul(torch.diag(theta), alpha), beta)), 
                        obs=params["M"])

def guide(params):
    
    num_samples = params["M"].size()[0]
    K = params["beta_fixed"].size()[0]

    with pyro.plate("K", K):
        with pyro.plate("N", num_samples):
            alpha = pyro.param("alpha", dist.Normal(params["alpha"], 1).sample())
            pyro.sample("activities", dist.Delta(alpha))

