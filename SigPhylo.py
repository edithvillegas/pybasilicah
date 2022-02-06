import json
import numpy as np
import pandas as pd
import os
import torch
import pyro
import pyro.distributions as dist
import svi
import transfer
import utilities
import shutil
import csv

def inference(input):

    #------------------------------TEST
    data = {}
    alpha_list = []
    beta_list = []
    alpha_batch = pd.DataFrame()
    beta_batch = pd.DataFrame()
    likelihoods = []
    #------------------------------TEST

    # parameters dictionary
    params = {
        "folder"            : input["folder"],
        "M"                 : utilities.M_csv2tensor(input["M_path"]),
        "beta_fixed"        : utilities.beta_csv2tensor(input["beta_fixed_path"]), 
        #"beta_fixed"        : utilities.beta_name2tensor("/home/azad/Documents/thesis/SigPhylo/data/real/beta_list.txt"),
        "A"                 : utilities.A_csv2tensor(input["A_path"]),
        "k_denovo"          : input["k_denovo"], 
        "lambda"            : input["hyper_lambda"],
        "lr"                : 0.05,
        "steps_per_iter"    : 500, 
        "max_iter"          : 5,
        "epsilon"           : 0.0001
        }

    num_samples = params["M"].size()[0]
    K_fixed = params["beta_fixed"].size()[0]
    K_denovo = params["k_denovo"]
    theta = torch.sum(params["M"], axis=1)

    # create a directory (if exist overwrite)
    new_dir = "data/results/" + params["folder"] + "/K_" + str(params["k_denovo"]) + "_L_" + str(params["lambda"])
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir)

    #####################################################################################
    # step 0 : independent inference
    #####################################################################################
    print("iteration ", 0)

    # ====== variational parameters initialization ===================================
    params["alpha_init"] = dist.Normal(torch.zeros(num_samples, K_denovo + K_fixed), 1).sample()
    params["beta_init"] = dist.Normal(torch.zeros(K_denovo, 96), 1).sample()

    # ====== model priors initialization =============================================
    params["alpha"] = dist.Normal(torch.zeros(num_samples, K_denovo + K_fixed), 1).sample()
    params["beta"] = dist.Normal(torch.zeros(K_denovo, 96), 1).sample()

    svi.inference(params)

    # ====== update variational parameters initialization ========================
    params["alpha_init"] = pyro.param("alpha").clone().detach()
    params["beta_init"] = pyro.param("beta").clone().detach()

    # ====== update model priors =================================================
    params["alpha"] = pyro.param("alpha").clone().detach()
    params["beta"] = pyro.param("beta").clone().detach()

    # ====== alpha & beta batches ==============================================
    current_alpha, current_beta = utilities.get_alpha_beta(params)
    alpha_batch = utilities.alpha_batch_df(alpha_batch, current_alpha)
    likelihoods = utilities.likelihoods(params, likelihoods)

    #------------------------------TEST
    alpha_list.append(np.array(current_alpha))
    beta_list.append(np.array(current_beta))
    #------------------------------TEST

    #####################################################################################
    # step 1 : inference using transition matrix (iterations)
    #####################################################################################
    for i in range(params["max_iter"]):
        print("iteration ", i + 1)

        # ====== calculate transfer coeff ======================================
        transfer_coeff = transfer.calculate_transfer_coeff(params)

        # ====== update alpha prior with transfer coeff ========================
        params["alpha"] = torch.matmul(transfer_coeff, params["alpha"])

        # ====== do inference with updated alpha_prior and beta_prior ==========
        svi.inference(params)

        # ====== update variational parameters initialization ========================
        params["alpha_init"] = pyro.param("alpha").clone().detach()
        params["beta_init"] = pyro.param("beta").clone().detach()

        # ====== update model priors =====================================
        params["alpha"] = pyro.param("alpha").clone().detach()
        params["beta"] = pyro.param("beta").clone().detach()

        # ====== append infered parameters =====================================
        previous_alpha, previous_beta = current_alpha, current_beta
        current_alpha, current_beta = utilities.get_alpha_beta(params)
        alpha_batch = utilities.alpha_batch_df(alpha_batch, current_alpha)
        likelihoods = utilities.likelihoods(params, likelihoods)

        #------------------------------TEST
        alpha_list.append(np.array(current_alpha))
        beta_list.append(np.array(current_beta))
        #------------------------------TEST

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

    # lambda
    with open(new_dir + '/lambda.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow([params["lambda"]])

    # alpha
    alpha_np = np.array(current_alpha)
    alpha_df = pd.DataFrame(alpha_np)
    alpha_df.to_csv(new_dir + '/alpha.csv', index=False, header=False)

    # beta denovo
    mutation_features = utilities.beta_mutation_features(input["beta_fixed_path"])   # dtype:list
    signature_names = []
    for g in range(current_beta.size()[0]):
        signature_names.append("Unknown")
    beta_np = np.array(current_beta)
    beta_df = pd.DataFrame(beta_np, index=signature_names, columns=mutation_features)
    beta_df.to_csv(new_dir + '/beta.csv', index=True, header=True)

    # alphas over iterations
    alpha_batch.to_csv(new_dir + '/alphas.csv', index=False, header=False)

    # betas over iterations
    beta_batch.to_csv(new_dir + '/betas.csv', index=False, header=False)

    # likelihoods
    with open(new_dir + '/likelihoods.csv', 'w') as f:
        write = csv.writer(f)
        itr = len(likelihoods)
        for w in range(itr):
            write.writerow([w, likelihoods[w]])

    # reconstructed phylogeny
    beta = torch.cat((params["beta_fixed"], current_beta), axis=0)
    M_r = torch.matmul(torch.matmul(torch.diag(theta), current_alpha), beta)
    M_np = np.array(M_r)
    M_np_int = np.rint(M_np)
    M_df = pd.DataFrame(M_np_int, columns=mutation_features)
    M_df.to_csv(new_dir + '/M_r.csv', index=False, header=True)

    # cosine similarity (phylogeny vs reconstructed phylogeny)
    with open(new_dir + '/cosines.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow([utilities.cosine_sim(params["M"], M_r)])

    #------------------------------TEST
    data = {
        "k_denovo": params["k_denovo"], 
        "lambda": params["lambda"], 
        "likelihoods": likelihoods, 
        "alphas": alpha_list, 
        "betas": beta_list,
        "M_r": M_np_int,
        "cosine": utilities.cosine_sim(params["M"], M_r)
        }
    #------------------------------TEST


    return data, likelihoods[-1]
