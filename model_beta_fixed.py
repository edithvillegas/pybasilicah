import pandas as pd
import torch
import pyro
import pyro.distributions as dist
import aux

def model(M, params):
    
    num_samples = M.size()[0]
    K = params["beta"].size()[0]
    theta = torch.sum(M, axis=1)

    # parametrize the activity matrix as theta*alpha, where
    # theta encodes the total number of mutations of the branches
    # alpha's are percentages of signature activity

    # sample alpha from a normal distribution using alpha prior
    with pyro.plate("K tot", K):
        with pyro.plate("N", num_samples):
            alpha = pyro.sample("activities", dist.Normal(params["alpha"], 1))

    # enforce non negativity
    alpha = torch.exp(alpha)
    #beta_denovo = torch.exp(beta_denovo)

    # normalize
    alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))
    #beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))

    # build full beta matrix
    #beta = torch.cat((beta_fixed, beta_denovo), axis=0)

    my_path = "/home/azad/Documents/thesis/SigPhylo/data/"
    beta_file = "expected_beta.csv"
    # load data
    beta_full = pd.read_csv(my_path + beta_file)
    beta, signature_names, contexts = aux.get_signature_profile(beta_full)


    # write the likelihood
    with pyro.plate("context", 96):
        with pyro.plate("sample", num_samples):
            pyro.sample("obs", dist.Poisson(torch.matmul(torch.matmul(torch.diag(theta), alpha), beta)), obs=M)

def guide(M, params):
    
    num_samples = M.size()[0]
    K = params["beta"].size()[0]

    with pyro.plate("K tot", K):
        with pyro.plate("N", num_samples):
            alpha = pyro.param("alpha", dist.Normal(params["alpha"],1).sample())
            pyro.sample("activities", dist.Delta(alpha))

