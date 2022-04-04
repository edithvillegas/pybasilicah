import batch
import torch
import utilities
import BaySig.run as run
import numpy as np
import random
import pandas as pd
import pyro.distributions as dist
import pyro
import torch.nn.functional as F
import svi


#----------------------------------------------------------------------------------------------

def single_k_run(params):

    M = params["M"]
    num_samples = params["M"].size()[0]
    k_fixed = params["beta_fixed"].size()[0]
    k_denovo = params["k_denovo"]

    #data = {}   # initialize JSON file (output data)
    #print("| k_denovo =", k_denovo, "| Start Running")
    
    #----- variational parameters initialization ----------------------------------------OK
    params["alpha_init"] = dist.Normal(torch.zeros(num_samples, k_denovo + k_fixed), 1).sample()
    params["beta_init"] = dist.Normal(torch.zeros(k_denovo, 96), 1).sample()

    #----- model priors initialization --------------------------------------------------OK
    params["alpha"] = dist.Normal(torch.zeros(num_samples, k_denovo + k_fixed), 1).sample()
    params["beta"] = dist.Normal(torch.zeros(k_denovo, 96), 1).sample()

    svi.inference(params)

    #----- update model priors initialization -------------------------------------------OK
    params["alpha"] = pyro.param("alpha").clone().detach()
    params["beta"] = pyro.param("beta").clone().detach()

    #----- get alpha & beta -------------------------------------------------------------OK
    alpha, beta = utilities.get_alpha_beta(params)

    #----- calculate & save likelihood (list) -------------------------------------------OK
    lh = utilities.log_likelihood(params)

    #----- calculate & save BIC (list) --------------------------------------------------OK
    bic = utilities.BIC(params)

    #----- phylogeny reconstruction -----------------------------------------------------OK
    M_R = utilities.Reconstruct_M(params)

    '''
    #----- save as dictioary ------------------------------------------------------------
    data = {
        "k_denovo": k_denovo, 
        "alpha": np.array(alpha), 
        "beta": np.array(beta), 
        "log-like": lh, 
        "BIC": bic, 
        "M_R": np.rint(np.array(M_R)), 
        "cosine": F.cosine_similarity(M, M_R).tolist()
        }
    '''

    #return data
    return bic, alpha, beta

#----------------------------------------------------------------------------------------------

def multi_k_run(params, k_list):
    bestBIC = 10000000000
    bestK = -1
    for k in k_list:
        params["k_denovo"] = k
        bic, alpha, beta = single_k_run(params)
        #print(bic)
        if bic <= bestBIC:
            bestBIC = bic
            bestK = k
            betsAlpha = alpha
            bestBeta = beta
    return bestK, bestBIC, betsAlpha, bestBeta

#----------------------------------------------------------------------------------------------

def input_generator(num_samples, k_fixed, k_denovo, cosmic_path):

    # Target Data ------------------------------------------------------------------- 
    # create random simulated data as target (all in dataframe format)
    M, alpha, beta_fixed, beta_denovo = utilities.generate_data(num_samples, k_fixed, k_denovo)

    # Test Data ---------------------------------------------------------------------
    # create fixed signature list as test data input
    # all cosmic signatures excluded target data signatures (fixed + denovo)
    cosmic_signature_names, _, _ = utilities.beta_read_csv(cosmic_path) # list
    beta_names_list = list(beta_fixed.index) + list(beta_denovo.index)       # list
    for signature in beta_names_list:
        cosmic_signature_names.remove(signature)
    
    overlap = random.randint(0, k_fixed)    # no. of common signatures
    exceed = random.randint(0, k_fixed)     # no. of different signatures

    overlap_sig = random.sample(list(beta_fixed.index), k=overlap)  # common signatures list
    exceed_sig = random.sample(cosmic_signature_names, k=exceed)    # different signatures list

    beta_fixed_test_names, _, beta_fixed_test = utilities.beta_read_name(overlap_sig + exceed_sig, cosmic_path)

    # M                     : dataframe
    # alpha                 : dataframe
    # beta_fixed            : dataframe
    # beta_denovo           : dataframe
    # beta_fixed_names_test : list
    # beta_fixed_test       : torch.Tensor

    return M, alpha, beta_fixed, beta_denovo, beta_fixed_test_names, beta_fixed_test



def func(num_samples, k_fixed, k_denovo, k_list):

    #--------------------------------------------------------------------------------
    # Run Model
    #--------------------------------------------------------------------------------
    
    print("----------------------------------")
    print("expected alpha :\n", alpha)
    print("----------------------------------")
    print("best k (BIC) :", bestK)
    print("----------------------------------")
    print("inferred alpha :\n", pd.DataFrame(np.array(alpha_inf)))
    print("----------------------------------")

    a = (torch.sum(alpha_inf, axis=0) / np.array(alpha_inf).shape[0]).tolist()
    print("a:", a)
    b = [x for x in a if x <= 0.05]
    print("b:", b, "\n")
    #print("----------------------------------\n")

    print("overlap:", overlap)
    print("exceed:", exceed)

    print("real fixed :", list(beta_fixed.index))
    print("real denovo :", list(beta_denovo.index))
    print("----------------------------------")
    print("test fixed :", test_sig, "\n")
    print("----------------------------------")

    excluded = []
    if len(a)==0:
        print("all signatures are included!")
    else:
        for i in b: 
            index = a.index(i)
            excluded.append(test_sig[index])
            print("Signature", test_sig[index], "is not included!")
    

    if exceed == len(b):
        print("good job! Sir")
    else:
        print("such a bloody shit!")


#--------------------------------------------------------------------------------
# Run Model
#--------------------------------------------------------------------------------

num_samples = 5
k_fixed = 3
k_denovo = 1
k_list = [1, 2, 3, 4, 5]
cosmic_path = "/home/azad/Documents/thesis/SigPhylo/cosmic/cosmic_catalogue.csv"

M, alpha, beta_fixed, beta_denovo, beta_fixed_test_names, beta_fixed_test = input_generator(num_samples, k_fixed, k_denovo, cosmic_path)

params = {
    "M"                 : torch.tensor(M.values).float(), 
    "beta_fixed"        : beta_fixed_test, 
    "lr"                : 0.05, 
    "steps_per_iter"    : 500, 
    "cosmic_path"       : cosmic_path
    }

bestK, bestBIC, betsAlpha, bestBeta = multi_k_run(params, k_list)

for i in range(100):
    func(num_samples=5, k_fixed=3, k_denovo=1, k_list=[1, 2, 3, 4, 5])