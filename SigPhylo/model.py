def model_single_run(M,params):

    num_samples = M.size()[0]

    beta_fixed = params["beta_fixed"]

    K_fixed = beta_fixed.size()[0]

    K_denovo = params["k_denovo"]

    theta = torch.sum(M, axis=1)

    # parametrize the activity matrix as theta*alpha, where theta encodes the total number of mutations of the branches
    # and alpha's are percentages of signature activity

    # sample alpha from a normal distribution using alpha prior
    
    with pyro.plate("K tot", K_denovo + K_fixed):

        with pyro.plate("N", num_samples):

            alpha = pyro.sample("activities", dist.Normal(torch.zeros(num_samples,K_denovo + K_fixed), 1))

    # sample the extra signature profiles from normal distribution

    with pyro.plate("contexts", 96):

        with pyro.plate("K denovo", K_denovo):

            beta_denovo = pyro.sample("extra_signatures", dist.Normal(torch.zeros(K_denovo,96), 1))

    # enforce non negativity

    alpha = torch.exp(alpha)

    beta_denovo = torch.exp(beta_denovo)

    # normalize

    alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))

    beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))

    # build full beta matrix

    beta = torch.cat((beta_fixed, beta_denovo), axis=0)

    # write the likelihood

    with pyro.plate("context", 96):

        with pyro.plate("sample", num_samples):

            pyro.sample("obs", dist.Poisson(torch.matmul(torch.matmul(torch.diag(theta), alpha), beta)), obs=M)


def guide_single_run(M, params):

    num_samples = M.size()[0]

    K_fixed = params["beta_fixed"].size()[0]

    K_denovo = params["k_denovo"]

    with pyro.plate("K tot", K_denovo + K_fixed):

        with pyro.plate("N", num_samples):

            alpha = pyro.param("alpha", params["alpha_prior"])

            pyro.sample("activities", dist.Delta(alpha))

    with pyro.plate("contexts", 96):

        with pyro.plate("K denovo", K_denovo):

            beta = pyro.param("beta", params["beta_prior"])

            pyro.sample("extra_signatures", dist.Delta(beta))

