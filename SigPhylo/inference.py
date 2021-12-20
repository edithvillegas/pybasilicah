
def inference_single_run(M, params, lr=0.05, num_steps=200):

    pyro.clear_param_store()  # always clear the store before the inference

    # learning global parameters

    adam_params = {"lr": lr}
    optimizer = Adam(adam_params)
    elbo = Trace_ELBO()

    svi = SVI(model_single_run, guide_single_run, optimizer, loss=elbo)

#   inference

#   do gradient steps

    for step in range(num_steps):

        loss = svi.step(M, params)


def calculate_transfer_coeff(M,params):

    alpha = torch.exp(params["alpha_prior"])
    alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))

    beta_denovo = torch.exp(params["beta_prior"])
    beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))

    beta_fixed = params["beta_fixed"]

    A = params["A"]

    hyper_lambda = params["lambda"]

    beta = torch.cat((beta_fixed, beta_denovo), axis=0)

    theta = torch.sum(M, axis=1)

    num_samples = M.size()[0]

    cos = torch.zeros(num_samples, num_samples)

    for i in range(num_samples):

        for j in range(num_samples):

            if A[i, j] == 1:

                M_r = theta[i] * torch.matmul(alpha[j], beta)

                if i==j:

                    cos[i, j] = (1-hyper_lambda)*torch.dot(M[i], M_r)/(torch.norm(M[i])*torch.norm(M_r))

                else:

                    cos[i, j] = hyper_lambda*torch.dot(M[i],M_r)/(torch.norm(M[i])*torch.norm(M_r))

    w = cos / (torch.sum(cos, 1).unsqueeze(-1))

    return w


def full_inference(M, params, lr=0.05, steps_per_iteration=200, num_iterations=10):
    # first indipendent run

    num_samples = M.size()[0]

    K_fixed = params["beta_fixed"].size()[0]

    K_denovo = params["k_denovo"].size()[0]

    # step 0 : indipendent inference

    print("iteration ", 0)

    alpha_prior = dist.Normal(torch.zeros(num_samples, K_denovo + K_fixed), 1).sample()

    beta_prior = dist.Normal(torch.zeros(K_denovo, 96), 1).sample()

    params["alpha_prior"] = alpha_prior

    params["beta_prior"] = beta_prior

    inference_single_run(M, params, lr=lr,num_steps=steps_per_iteration)

    # do iterations transferring alpha's

    for i in range(num_iterations):

        print("iteration ", i + 1)

        # extract inferred alpha's and beta'a from pyro store

        params["alpha_prior"] = torch.tensor(pyro.param("alpha"))

        params["beta_prior"] = torch.tensor(pyro.param("beta"))

        # calculate transfer coeff

        transfer_coeff = calculate_transfer_coeff(M,params)

        # update alpha prior with transfer coeff

        params["alpha_prior"] = torch.matmul(transfer_coeff,params["alpha_prior"])

        # do inference with updates alpha_prior and beta_prior

        inference_single_run(M, params, lr=lr,num_steps=steps_per_iteration)

    # save final inference

    params["alpha_prior"] = torch.tensor(pyro.param("alpha"))

    params["beta_prior"] = torch.tensor(pyro.param("beta"))

    alpha_denovo = torch.exp(params["alpha_prior"])
    alpha_denovo = alpha_denovo / (torch.sum(alpha_denovo, 1).unsqueeze(-1))

    params["alpha_denovo"] = alpha_denovo

    beta_denovo = torch.exp(params["beta_prior"])
    beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))

    params["beta_denovo"] = beta_denovo

    return params