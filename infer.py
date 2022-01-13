import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist

import svi
import transfer
import utilities


def full_inference(M, params, lr=0.05, steps_per_iteration=200, max_num_iterations=100):

    num_samples = M.size()[0]
    K_fixed = params["beta_fixed"].size()[0]
    K_denovo = params["k_denovo"]

    # ====== initialize alpha and beta tracking list ===========================
    alphas = []
    betas = []

    #####################################################################################
    ###### step 0 : independent inference ####################################################
    #####################################################################################
    print("iteration ", 0)

    params["alpha"] = dist.Normal(torch.zeros(num_samples, K_denovo + K_fixed), 1).sample()
    params["beta"] = dist.Normal(torch.zeros(K_denovo, 96), 1).sample()

    #alpha = dist.Normal(torch.zeros(num_samples, K_denovo + K_fixed), 1).sample()
    #beta = dist.Normal(torch.zeros(K_denovo, 96), 1).sample()

    svi.single_inference(M, params, lr=lr, num_steps=steps_per_iteration)

    # ====== update infered parameters =========================================
    params["alpha"] = pyro.param("alpha").clone().detach()
    params["beta"] = pyro.param("beta").clone().detach()

    # ====== append infered parameters =========================================
    a, b = utilities.get_alpha_beta2(pyro.param("alpha").clone().detach(), pyro.param("beta").clone().detach())
    alphas.append(a)
    betas.append(b)
    a_np = np.array(a)
    a_df = pd.DataFrame(a_np)
    a_df.to_csv('results/alphas.csv', index=False, header=False)
    b_np = np.array(b)
    b_df = pd.DataFrame(b_np)
    b_df.to_csv('results/betas.csv', index=False, header=False)

    #####################################################################################
    ###### step 1 : iterations (transferring alpha) #####################################
    #####################################################################################
    for i in range(max_num_iterations):
        ind = 1 # 1 means stop iterations
        print("iteration ", i + 1)

        # ====== calculate transfer coeff ======================================
        transfer_coeff = transfer.calculate_transfer_coeff(M, params)

        # ====== update alpha prior with transfer coeff ========================
        params["alpha"] = torch.matmul(transfer_coeff, params["alpha"])

        # ====== do inference with updated alpha_prior and beta_prior ==========
        svi.single_inference(M, params, lr=lr, num_steps=steps_per_iteration)

        # ====== update infered parameters =====================================
        params["alpha"] = pyro.param("alpha").clone().detach()
        params["beta"] = pyro.param("beta").clone().detach()

        # ====== append infered parameters =====================================
        a, b = utilities.get_alpha_beta2(pyro.param("alpha").clone().detach(), pyro.param("beta").clone().detach())
        alphas.append(a)
        betas.append(b)
        a_np = np.array(a)
        a_df = pd.DataFrame(a_np)
        a_df.to_csv('results/alphas.csv', index=False, header=False, mode='a')
        b_np = np.array(b)
        b_df = pd.DataFrame(b_np)
        b_df.to_csv('results/betas.csv', index=False, header=False, mode='a')
        
        # ====== likelihood ====================================================
        theta = torch.sum(M, axis=1)
        b_full = torch.cat((params["beta_fixed"], b), axis=0)
        # take care of it later
        lh = dist.Poisson(torch.matmul(torch.matmul(torch.diag(theta), a), b_full)).log_prob(M)
        #print("likelihood :", lh)

        # ====== error =========================================================
        #loss_alpha = torch.sum((alphas[i] - alphas[i+1]) ** 2)
        #loss_beta = torch.sum((betas[i] - betas[i+1]) ** 2)
        #print("loss alpha =", loss_alpha)
        #print("loss beta =", loss_beta)
        
        # ====== convergence test ==============================================
        eps = 0.05
        for j in range(num_samples):
            for k in range(K_fixed + K_denovo):
                ratio = alphas[-1][j][k].item() / alphas[-2][j][k].item()
                if (ratio > 1 + eps or ratio < 1 - eps ):
                #if torch.abs(current[j][k].item() - previous[j][k]).item() > epsilon:
                    ind = 0

        if (ind == 1):
            print("meet convergence criteria, stoped in iteration", i+1)
            break


    return params, alphas, betas
