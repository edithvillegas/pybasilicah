import torch
import pyro
import pyro.distributions as dist
import svi
import transfer
import aux


def full_inference(M, params, lr=0.05, steps_per_iteration=200, num_iterations=10):
    
    # first indipendent run

    num_samples = M.size()[0]
    K_fixed = params["beta_fixed"].size()[0]
    K_denovo = params["k_denovo"]

    # create a list of alphas and betas in each step of the iterations
    alphas = []
    betas = []


    # step 0 : independent inference

    print("iteration ", 0)
    params["alpha"] = dist.Normal(torch.zeros(num_samples, K_denovo + K_fixed), 1).sample()
    params["beta"] = dist.Normal(torch.zeros(K_denovo, 96), 1).sample()

    svi.single_inference(M, params, lr=lr, num_steps=steps_per_iteration)

    # add the alpha and beta from step zero to the list
    a, b = aux.get_alpha_beta2(pyro.param("alpha").clone().detach(), pyro.param("beta").clone().detach())
    alphas.append(a)
    betas.append(b)


    # do iterations using transferring alpha's

    for i in range(num_iterations):

        print("iteration ", i + 1)
        params["alpha"] = pyro.param("alpha").clone().detach()
        params["beta"] = pyro.param("beta").clone().detach()

        # calculate transfer coeff
        transfer_coeff = transfer.calculate_transfer_coeff(M, params)

        # update alpha prior with transfer coeff
        params["alpha"] = torch.matmul(transfer_coeff, params["alpha"])

        # do inference with updated alpha_prior and beta_prior
        svi.single_inference(M, params, lr=lr, num_steps=steps_per_iteration)

        # add the new alpha and beta to the list
        a, b = aux.get_alpha_beta2(pyro.param("alpha").clone().detach(), pyro.param("beta").clone().detach())
        alphas.append(a)
        betas.append(b)

        loss_alpha = torch.sum((alphas[i] - alphas[i+1]) ** 2)
        loss_beta = torch.sum((betas[i] - betas[i+1]) ** 2)

        #print("loss alpha =", loss_alpha)
        #print("loss beta =", loss_beta)

    # save final inference
    params["alpha"] = pyro.param("alpha").clone().detach()
    params["beta"] = pyro.param("beta").clone().detach()

    return params, alphas, betas
