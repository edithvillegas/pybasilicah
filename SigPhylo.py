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
import torch.nn.functional as F

def single_run(params):

    dir = params["dir"]
    M = params["M"]
    num_samples = params["M"].size()[0]
    k_fixed = params["beta_fixed"].size()[0]
    k_denovo = params["k_denovo"]
    landa = params["lambda"]
    mutation_features = params["mutation_features"]
    max_iter = params["max_iter"]

    #------------------------------TEST

    #-----JSON-----
    data = {}

    alpha_list = []
    beta_list = []

    #-----CSV-----
    alpha_batch = pd.DataFrame()

    # list of likelihoods over iterations
    LHs_over_iters = []
    #------------------------------TEST

    #------------------------------------------------------------------------------------
    #----- create directory (if exist overwrite) ----------------------------------------
    #------------------------------------------------------------------------------------
    sub_dir = dir + "/K_" + str(k_denovo) + "_L_" + str(landa)
    if os.path.exists(sub_dir):
        shutil.rmtree(sub_dir)
    os.makedirs(sub_dir)

    #====================================================================================
    # step 0 : independent inference ----------------------------------------------------
    #====================================================================================
    print("iteration ", 0)

    #----- variational parameters initialization ----------------------------------------OK
    params["alpha_init"] = dist.Normal(torch.zeros(num_samples, k_denovo + k_fixed), 1).sample()
    params["beta_init"] = dist.Normal(torch.zeros(k_denovo, 96), 1).sample()

    #----- model priors initialization --------------------------------------------------OK
    params["alpha"] = dist.Normal(torch.zeros(num_samples, k_denovo + k_fixed), 1).sample()
    params["beta"] = dist.Normal(torch.zeros(k_denovo, 96), 1).sample()

    svi.inference(params)

    #----- update variational parameters initialization ---------------------------------OK
    params["alpha_init"] = pyro.param("alpha").clone().detach()
    params["beta_init"] = pyro.param("beta").clone().detach()

    #----- update model priors initialization -------------------------------------------OK
    params["alpha"] = pyro.param("alpha").clone().detach()
    params["beta"] = pyro.param("beta").clone().detach()

    #----- get alpha & beta -------------------------------------------------------------OK
    current_alpha, current_beta = utilities.get_alpha_beta(params)

    #----- TEST -------------------------------------------------------------------------
    alpha_batch = utilities.alpha_batch_df(alpha_batch, current_alpha)
    alpha_list.append(np.array(current_alpha))
    beta_list.append(np.array(current_beta))

    #----- calculate & save likelihood (list) -------------------------------------------OK
    LH = utilities.likelihood(params)
    LHs_over_iters.append(LH)

    #====================================================================================
    # step 1 : inference using transition matrix (iterations)
    #====================================================================================
    for i in range(max_iter):
        print("iteration ", i + 1)

        #----- calculate transfer coeff -------------------------------------------------OK
        transfer_coeff = transfer.calculate_transfer_coeff(params)

        #----- update alpha prior with transfer coeff -----------------------------------OK
        params["alpha"] = torch.matmul(transfer_coeff, params["alpha"])

        svi.inference(params)

        #----- update variational parameters initialization -----------------------------OK
        params["alpha_init"] = pyro.param("alpha").clone().detach()
        params["beta_init"] = pyro.param("beta").clone().detach()

        #----- update model priors initialization ---------------------------------------OK
        params["alpha"] = pyro.param("alpha").clone().detach()
        params["beta"] = pyro.param("beta").clone().detach()

        #----- update alpha & beta ------------------------------------------------------OK
        previous_alpha, previous_beta = current_alpha, current_beta
        current_alpha, current_beta = utilities.get_alpha_beta(params)

         #----- TEST --------------------------------------------------------------------
        alpha_batch = utilities.alpha_batch_df(alpha_batch, current_alpha)
        alpha_list.append(np.array(current_alpha))
        beta_list.append(np.array(current_beta))

        #----- calculate & save likelihood (list) ---------------------------------------OK
        LH = utilities.likelihood(params)
        LHs_over_iters.append(LH)
        
        #----- convergence test ---------------------------------------------------------
        if (utilities.convergence(current_alpha, previous_alpha, params) == "stop"):
            print("meet convergence criteria, stoped in iteration", i+1)
            break

    #====================================================================================
    # write to CSV file -----------------------------------------------------------------
    #====================================================================================

    #----- final alpha ------------------------------------------------------------------OK
    a_np = np.array(current_alpha)
    a_df = pd.DataFrame(a_np)
    a_df.to_csv(sub_dir + '/alpha.csv', index=False, header=False)

    #----- final beta denovo ------------------------------------------------------------OK
    b_np = np.array(current_beta)
    b_df = pd.DataFrame(b_np, index = k_denovo * ["Unknown"], columns = mutation_features)
    b_df.to_csv(sub_dir + '/beta.csv', index=True, header=True)

    #----- likelihoods list -------------------------------------------------------------OK
    with open(sub_dir + '/likelihoods.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(LHs_over_iters)

    #----- alphas over iterations -------------------------------------------------------
    alpha_batch.to_csv(sub_dir + '/alphas.csv', index=False, header=False)

    #----- phylogeny reconstruction -----------------------------------------------------OK
    M_r = utilities.Reconstruct_M(params)
    M_np = np.array(M_r)
    M_np_int = np.rint(M_np)
    M_df = pd.DataFrame(M_np_int, columns=mutation_features)
    M_df.to_csv(sub_dir + '/M_r.csv', index=False, header=True)

    #----- cosine similarity (original vs reconstructed) --------------------------------OK
    with open(sub_dir + '/cosines.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(F.cosine_similarity(M, M_r).tolist())

    #----- data as dictioary ------------------------------------------------------------
    data = {
        "k_denovo": k_denovo, 
        "lambda": landa, 
        "alpha": current_alpha, 
        "beta": current_beta, 
        "likelihoods": LHs_over_iters, 
        "M_r": np.rint(np.array(M_r)), 
        "cosine": F.cosine_similarity(M, M_r).tolist()
        }

    #return data, LHs_over_iters[-1]
    return data



#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

#"beta_fixed"        : utilities.beta_read_name(["SBS5"])[2],

input = {
    "M_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv", 
    "beta_fixed_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv", 
    "A_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/A.csv", 
    "K_list" : [1, 2, 3], 
    "lambda_list" : [0, 0.5, 1], 
    "dir" : "/home/azad/Documents/thesis/SigPhylo/data/results/output"
}

def batch_run(input):

    # create new directory (overwrite if exist)
    new_dir = input["dir"]
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir)

    params = {
        "M"                 : utilities.M_read_csv(input["M_path"])[1], 
        "mutation_features" : utilities.beta_read_csv(input["beta_fixed_path"])[1], 
        "beta_fixed"        : utilities.beta_read_csv(input["beta_fixed_path"])[2], 
        "A"                 : utilities.A_read_csv(input["A_path"]), 
        "lr"                : 0.05, 
        "steps_per_iter"    : 500, 
        "max_iter"          : 100, 
        "epsilon"           : 0.0001, 
        "dir"               : input["dir"]
        }


    likelihoods = []
    Data = {}
    Data["M"] = np.array(params["M"])
    Data["beta_fixed"] = np.array(params["beta_fixed"])
    Data["A"] = np.array(params["A"])
    i = 1
    
    for k in input["k_list"]:
        for landa in input["lambda_list"]:

            print("k_denovo =", k, "| lambda =", landa)

            params["k_denovo"] = k
            params["lambda"] = landa

            data, L = single_run(params)

            likelihoods.append(L)
            #encodedNumpyData = json.dumps(data, cls=utilities.NumpyArrayEncoder)

            # likelihoods over lambdas
            with open(new_dir + "/likelihoods.csv", 'a') as f:
                write = csv.writer(f)
                write.writerow([k, landa, L])

            Data[str(i)] = data
            i += 1

    Data["likelihoods"] = likelihoods

    with open(new_dir + "/output.json", 'w') as outfile:
        json.dump(Data, outfile, cls=utilities.NumpyArrayEncoder)


'''
#----------------------------------------------------------------------------
#------------------------------ STORAGE -------------------------------------
#----------------------------------------------------------------------------
# betas over iterations
beta_batch = pd.DataFrame()
beta_batch.to_csv(sub_dir + '/betas.csv', index=False, header=False)

# lambda
with open(sub_dir + '/lambda.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow([landa])

#----- error --------------------------------------------------------------------
loss_alpha = torch.sum((alphas[i] - alphas[i+1]) ** 2)
loss_beta = torch.sum((betas[i] - betas[i+1]) ** 2)
print("loss alpha =", loss_alpha)
print("loss beta =", loss_beta)

'''



