import torch

def calculate_transfer_coeff(params):
    
    alpha = torch.exp(params["alpha"])
    alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))

    beta_denovo = torch.exp(params["beta"])
    beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))

    beta_fixed = params["beta_fixed"]

    A = params["A"]

    hyper_lambda = params["lambda"]

    beta = torch.cat((beta_fixed, beta_denovo), axis=0)

    theta = torch.sum(params["M"], axis=1)

    num_samples = params["M"].size()[0]

    cos = torch.zeros(num_samples, num_samples)

    for i in range(num_samples):
        for j in range(num_samples):

            if A[i, j] == 1:

                M_r = theta[i] * torch.matmul(alpha[j], beta)

                if i==j:
                    cos[i, j] = (1-hyper_lambda)*torch.dot(params["M"][i], M_r) / (torch.norm(params["M"][i])*torch.norm(M_r))
                else:
                    cos[i, j] = hyper_lambda*torch.dot(params["M"][i],M_r)/(torch.norm(params["M"][i])*torch.norm(M_r))

    w = cos / (torch.sum(cos, 1).unsqueeze(-1))

    return w