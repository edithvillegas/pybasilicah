import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
import svi
import transfer
import utilities


# write to CSV file ??????

def full_inference(input):

    # parameters dictionary
    params = {
        "M"                 : utilities.M_csv2tensor(input["M_path"]),
        "beta_fixed"        : utilities.beta_csv2tensor(input["beta_fixed_path"]), 
        "A"                 : utilities.A_csv2tensor(input["A_path"]),
        "k_denovo"          : input["k_denovo"], 
        "lambda"            : input["hyper_lambda"],
        "lr"                : input["lr"],
        "steps_per_iter"    : input["steps_per_iter"], 
        "max_iter"          : input["max_iter"],
        "epsilon"           : input["epsilon"]
        }

    num_samples = params["M"].size()[0]
    K_fixed = params["beta_fixed"].size()[0]
    K_denovo = params["k_denovo"]

    #####################################################################################
    ###### step 0 : independent inference ###############################################
    #####################################################################################
    print("iteration ", 0)

    params["alpha"] = dist.Normal(torch.zeros(num_samples, K_denovo + K_fixed), 1).sample()
    params["beta"] = dist.Normal(torch.zeros(K_denovo, 96), 1).sample()

    svi.single_inference(params)

    # ====== update infered parameters =========================================
    params["alpha"] = pyro.param("alpha").clone().detach()
    params["beta"] = pyro.param("beta").clone().detach()

    # ====== alpha & beta batches ==============================================
    current_alpha, current_beta = utilities.get_alpha_beta(params)
    alpha_list, beta_list = [], []
    alpha_list.append(current_alpha)
    beta_list.append(current_beta)

    #####################################################################################
    ###### step 1 : iterations (transferring alpha) #####################################
    #####################################################################################
    for i in range(params["max_iter"]):
        print("iteration ", i + 1)

        #utilities.alphas_betas_tensor2csv(params, append=0)

        # ====== calculate transfer coeff ======================================
        transfer_coeff = transfer.calculate_transfer_coeff(params)

        # ====== update alpha prior with transfer coeff ========================
        params["alpha"] = torch.matmul(transfer_coeff, params["alpha"])

        # ====== do inference with updated alpha_prior and beta_prior ==========
        svi.single_inference(params)

        # ====== update infered parameters =====================================
        params["alpha"] = pyro.param("alpha").clone().detach()
        params["beta"] = pyro.param("beta").clone().detach()

        # ====== append infered parameters =====================================
        previous_alpha, previous_beta = current_alpha, current_beta
        current_alpha, current_beta = utilities.get_alpha_beta(params)
        alpha_list.append(current_alpha)
        beta_list.append(current_beta)
        #utilities.alphas_betas_tensor2csv(params, append=1)
        
        # ====== likelihood ====================================================
        #theta = torch.sum(M, axis=1)
        #b_full = torch.cat((params["beta_fixed"], current_beta), axis=0)
        #lh = dist.Poisson(torch.matmul(torch.matmul(torch.diag(theta), current_alpha), b_full)).log_prob(M)
        #print("likelihood :", lh)

        # ====== error =========================================================
        #loss_alpha = torch.sum((alphas[i] - alphas[i+1]) ** 2)
        #loss_beta = torch.sum((betas[i] - betas[i+1]) ** 2)
        #print("loss alpha =", loss_alpha)
        #print("loss beta =", loss_beta)
        
        # ====== convergence test ==============================================
        if (utilities.convergence(current_alpha, previous_alpha, params) == "stop"):
            print("meet convergence criteria, stoped in iteration", i+1)
            break

    # ====== write to CSV file ==========================================

    alpha_batch = torch.stack(alpha_list)
    beta_batch = torch.stack(beta_list)
    
    #print(alpha_batch.shape)
    #print(beta_batch.shape)
    
    alpha_batch_np = np.array(alpha_batch)
    alpha_batch_df = pd.DataFrame(alpha_batch_np)
    alpha_batch_df.to_csv('results/alphas.csv', index=False, header=False)

    beta_batch_np = np.array(beta_batch)
    beta_batch_df = pd.DataFrame(beta_batch_np)
    beta_batch_df.to_csv('results/betas.csv', index=False, header=False)

    alpha_np = np.array(current_alpha)
    alpha_df = pd.DataFrame(alpha_np)
    alpha_df.to_csv('results/alpha.csv', index=False, header=False)

    beta_np = np.array(current_beta)
    beta_df = pd.DataFrame(beta_np)
    beta_df.to_csv('results/beta.csv', index=False, header=False)
