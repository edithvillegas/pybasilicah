import pyro
import pyro.distributions as dist

def guide(params):

    num_samples = params["M"].size()[0]
    K_fixed = params["beta_fixed"].size()[0]
    K_denovo = params["k_denovo"]

    with pyro.plate("K", K_denovo + K_fixed):
        with pyro.plate("N", num_samples):
            alpha = pyro.param("alpha", dist.Normal(params["alpha_init"], 1).sample())
            pyro.sample("activities", dist.Delta(alpha))

    with pyro.plate("contexts", 96):
        with pyro.plate("K_denovo", K_denovo):
            beta = pyro.param("beta", dist.Normal(params["beta_init"], 1).sample())
            pyro.sample("extra_signatures", dist.Delta(beta))
