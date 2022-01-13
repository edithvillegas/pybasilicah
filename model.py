import pyro
import torch
import pyro.distributions as dist


def model(M, params):
    '''
    ====== inputs ======
    * M --> dtype:torch.Tensor
    * params = {"alpha" : alpha, 
                "beta" : beta,
                "k_denovo" : k_denovo, 
                "beta_fixed" : beta_counts, 
                "A" : A, 
                "lambda": 0.9}
    '''
    
    num_samples = M.size()[0]
    beta_fixed = params["beta_fixed"]
    K_fixed = beta_fixed.size()[0]
    K_denovo = params["k_denovo"]
    theta = torch.sum(M, axis=1)

    # parametrize the activity matrix as theta*alpha
    # theta encodes the total number of mutations of the branches
    # alpha is relative exposure (percentages of signature activity)

    # sample alpha from a normal distribution using alpha prior
    with pyro.plate("K", K_denovo + K_fixed):   # columns
        with pyro.plate("N", num_samples):      # rows
            alpha = pyro.sample("activities", dist.Normal(params["alpha"], 1))

    # sample the extra signature profiles from normal distribution
    with pyro.plate("contexts", 96):
        with pyro.plate("K_denovo", K_denovo):
            beta_denovo = pyro.sample("extra_signatures", dist.Normal(params["beta"], 1))

    # enforce non negativity
    alpha = torch.exp(alpha)
    beta_denovo = torch.exp(beta_denovo)

    # normalize
    alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))
    beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))

    # build full signature profile (beta) matrix
    beta = torch.cat((beta_fixed, beta_denovo), axis=0)

    # compute the likelihood
    with pyro.plate("context", 96):
        with pyro.plate("sample", num_samples):
            pyro.sample("obs", 
                        dist.Poisson(torch.matmul(torch.matmul(torch.diag(theta), alpha), beta)), 
                        obs=M)
