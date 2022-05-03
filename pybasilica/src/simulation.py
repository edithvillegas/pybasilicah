import random
import pyro.distributions as dist
import numpy as np
import pandas as pd
import torch
from statistics import mean
import torch.nn.functional as F
import basilica
import utilities


#-----------------------------------------------------------------[<QC-PASSED>]
# divide cosmic signatures into two parts:
# 1st part: considered as cosmic signatures
# 2nd part: considered as denovo signatures

def cosmic_denovo(cosmic_path):
    # cosmic_path --- <class 'str'>
    full_cosmic_df = pd.read_csv(cosmic_path, index_col=0)
    full_cosmic_list = list(full_cosmic_df.index)

    cosmic_list = random.sample(full_cosmic_list, k=50)
    denovo_list = list(set(full_cosmic_list) - set(cosmic_list))

    cosmic_df = full_cosmic_df.loc[cosmic_list] # <class 'pandas.core.frame.DataFrame'>
    denovo_df = full_cosmic_df.loc[denovo_list] # <class 'pandas.core.frame.DataFrame'>

    return cosmic_df, denovo_df
    # cosmic_df --- dtype: DataFrame
    # denovo_df --- dtype: DataFrame

#-----------------------------------------------------------------[QC:PASSED]
def target_generator(profile, cosmic_df, denovo_df):
    # profile ------- {"A", "B", "C"}
    # cosmic_df ----- <class 'pandas.core.frame.DataFrame'>
    # denovo_df ----- <class 'pandas.core.frame.DataFrame'>

    num_samples = random.randint(15, 25)

    # error handling for profile argument
    valid = ["A", "B", "C"]
    if profile not in valid:
        raise ValueError("profile must be one of %s." % valid)

    if profile=="A":
        fixed_num = random.randint(3, 5)    # <class 'int'>
        denovo_num = random.randint(0, 2)
    elif profile=="B":
        fixed_num = random.randint(0, 2)
        denovo_num = random.randint(3, 5)
    elif profile=="C":
        fixed_num = random.randint(3, 5)
        denovo_num = random.randint(3, 5)

    cosmic_list = list(cosmic_df.index)
    denovo_list = list(denovo_df.index)
    mutation_features = list(cosmic_df.columns)

    # beta fixed ------------------------------------------------------
    if fixed_num > 0:
        fixed_list = random.sample(cosmic_list, k=fixed_num)
        beta_fixed_df = cosmic_df.loc[fixed_list]
    else:
        beta_fixed_df = pd.DataFrame(columns=mutation_features)
    
    # beta denovo -----------------------------------------------------
    if denovo_num > 0:
        denovo_list = random.sample(denovo_list, k=denovo_num)
        beta_denovo_df = denovo_df.loc[denovo_list]

        # add "_D" to the end of denovo signatures to distinguish them from fixed signatures
        denovo_labels = []
        for i in range(len(denovo_list)):
            denovo_labels.append(denovo_list[i] + '_D')
        beta_denovo_df.index = denovo_labels
    else:
        beta_denovo_df = pd.DataFrame(columns=mutation_features)


    if beta_denovo_df.empty:
        beta_df = beta_fixed_df
    elif beta_fixed_df.empty:
        beta_df = beta_denovo_df
    else:
        beta_df = pd.concat([beta_fixed_df, beta_denovo_df], axis=0)

    signatures = list(beta_df.index)
    beta_tensor = torch.tensor(beta_df.values).float()

    #------- alpha ----------------------------------------------
    matrix = np.random.rand(num_samples, len(signatures))
    alpha_tensor = torch.tensor(matrix / matrix.sum(axis=1)[:, None]).float()
    alpha_np = np.array(alpha_tensor)
    alpha_df = pd.DataFrame(alpha_np, columns=signatures)

    #------- theta ----------------------------------------------
    theta = random.sample(range(1000, 4000), k=num_samples) # dtype:list
    
    #------- check dimensions -----------------------------------
    m_alpha = alpha_tensor.size()[0]    # no. of branches (alpha)
    m_theta = len(theta)                # no. of branches (theta)
    k_alpha = alpha_tensor.size()[1]    # no. of signatures (alpha)
    k_beta = beta_tensor.size()[0]      # no. of signatures (beta)
    if not(m_alpha == m_theta and k_alpha == k_beta):
        raise ValueError("wrong dimensions!")
    
    M_tensor = torch.zeros([num_samples, 96])   # initialize mutational catalogue with zeros

    for i in range(num_samples):
        p = alpha_tensor[i]         # selecting branch i
        for k in range(theta[i]):   # iterate for number of the mutations in branch i

            # sample signature profile index from categorical data
            b = beta_tensor[dist.Categorical(p).sample().item()]

            # sample mutation feature index for corresponding signature from categorical data
            j = dist.Categorical(b).sample().item()

            # add +1 to the mutation feature in position j in branch i
            M_tensor[i, j] += 1

    # phylogeny
    M_np = np.array(M_tensor)
    M_df = pd.DataFrame(M_np, columns=mutation_features)
    M_df = M_df.astype(int)

    # return all in dataframe format
    return M_df, alpha_df, beta_fixed_df, beta_denovo_df
    # M_df ------------- dtype: dataframe
    # alpha_df --------- dtype: dataframe
    # beta_fixed_df ---- dtype: dataframe, could be empty (if empty --> beta_denovo_df != empty)
    # beta_denovo_df --- dtype: dataframe, could be empty (if empty --> beta_fixed_df != empty)

#-----------------------------------------------------------------[QC:PASSED]
def beta_input_generator(profile, beta_fixed_df, beta_denovo_df, cosmic_df):
    # profile ---------- {"X", "Y", "Z"}
    # beta_fixed_df ---- dtype: dataframe
    # beta_denovo_df --- dtype: dataframe
    # cosmic_df -------- dtype: dataframe

    # error handling for profile argument
    valid = ["X", "Y", "Z"]
    if profile not in valid:
        raise ValueError("profile must be one of %s." % valid)

    # TARGET DATA -----------------------------------------------------------------------------
    beta_fixed_names = list(beta_fixed_df.index)    # target beta signatures names (dtype: list)
    k_fixed_target = len(beta_fixed_names)
    beta_denovo_names = list(beta_denovo_df.index)  # target beta signatures names (dtype: list)
    k_denovo_target = len(beta_denovo_names)

    # common fixed signatures (target intersect test)
    # different fixed signatures (test minus target)
    if profile=="X":
        if k_fixed_target > 0:
            k_overlap = random.randint(1, k_fixed_target)
            k_extra = 0
        else:
            k_overlap = 0
            k_extra = random.randint(1, k_denovo_target)

    elif profile=="Y":
        if k_fixed_target > 0:
            k_overlap = random.randint(1, k_fixed_target)
            k_extra = random.randint(1, k_fixed_target)
        else:
            k_overlap = 0
            k_extra = random.randint(1, k_denovo_target)

    elif profile=="Z":
        k_overlap = 0
        k_extra = random.randint(1, k_fixed_target + k_denovo_target)
    
    
    cosmic_names = list(cosmic_df.index)    # cosmic signatures names (dtype: list)

    # exclude beta target fixed signatures from cosmic
    for signature in beta_fixed_names:
        cosmic_names.remove(signature)
    
    # common fixed signatures list
    if k_overlap > 0:
        overlap_sigs = random.sample(beta_fixed_names, k=k_overlap)
    else:
        overlap_sigs = []
    
    # different fixed signatures list
    if k_extra > 0:
        extra_sigs = random.sample(cosmic_names, k=k_extra)
    else:
        extra_sigs = []
    
    beta_input = cosmic_df.loc[overlap_sigs + extra_sigs]

    return beta_input   # dtype: dataframe


#-----------------------------------------------------------------[PASSED]
def input_generator(Tprofile, Iprofile, cosmic_df, denovo_df):

    M_df, alpha_df, beta_fixed_df, beta_denovo_df = target_generator(Tprofile, cosmic_df, denovo_df)
    beta_input_df = beta_input_generator(Iprofile, beta_fixed_df, beta_denovo_df, cosmic_df)
    #beta_df = pd.concat([beta_fixed_df, beta_denovo_df], axis=0)

    data = {
        "M" : M_df,                         # dataframe
        "alpha" : alpha_df,                 # dataframe
        "beta_fixed" : beta_fixed_df,       # dataframe
        "beta_denovo" : beta_denovo_df,     # dataframe
        "beta_input" : beta_input_df,       # dataframe
        "cosmic_df" : cosmic_df             # dataframe
    }

    return data



def run_simulated(Tprofile, Iprofile, cos_path_org, fixedLimit, denovoLimit, seed):
    
    random.seed(seed)

    try:

        # ========== INPUT ==========================================================
        cosmic_df, denovo_df = cosmic_denovo(cos_path_org)
        input_data = input_generator(Tprofile, Iprofile, cosmic_df, denovo_df)
        M = input_data["M"]                     # dataframe
        A = input_data["alpha"]                 # dataframe
        B_fixed = input_data["beta_fixed"]      # dataframe
        B_denovo = input_data["beta_denovo"]    # dataframe
        B_input = input_data["beta_input"]      # dataframe
        #B_input = cosmic_df
        cosmic_df = input_data["cosmic_df"]     # dataframe
        k_list = [0, 1, 2, 3, 4, 5]             # list

        '''
        print("========================================================")
        print("theta:", torch.sum(torch.tensor(data["M"].values), axis=1).float())
        print("Alpha Target\n",         A)
        print("Beta Fixed Target",    B_fixed)
        print("Beta Denovo Target",  B_denovo)
        print("Beta Input",           B_input)
        '''

        # ========== OUTPUT =========================================================
        A_inf, B_fixed_inf, B_denovo_inf = basilica.BaSiLiCa(M, B_input, k_list, cosmic_df, fixedLimit, denovoLimit) # all dataframe

        # ========== Metrics ========================================================
        
        B_fixed_accuracy = utilities.betaFixed_perf(B_input, B_fixed, B_fixed_inf)
        B_denovo_inf_labeled, B_denovo_quantity, B_denovo_quality = utilities.betaDenovo_perf(B_denovo_inf, B_denovo)

        theta_tensor = torch.sum(torch.tensor(M.values).float(), axis=1)
        A_tensor = torch.tensor(A_inf.values).float()
        if B_denovo_inf.empty:
            beta_df = B_fixed_inf
        else:
            beta_df = pd.concat([B_fixed_inf, B_denovo_inf], axis=0)
        B_tensor = torch.Tensor(beta_df.values).float()
        M_r = torch.matmul(torch.matmul(torch.diag(theta_tensor), A_tensor), B_tensor)
        gof = mean(F.cosine_similarity(torch.tensor(M.values).float(), M_r).tolist())
        

        output = {
            "M"                 : M,                    # dataframe
            "A_target"          : A,                    # dataframe
            "B_fixed_target"    : B_fixed,              # dataframe
            "B_denovo_target"   : B_denovo,             # dataframe
            "B_input"           : B_input,              # dataframe
            "A_inf"             : A_inf,                # dataframe
            "B_fixed_inf"       : B_fixed_inf,          # dataframe
            "B_denovo_inf"      : B_denovo_inf,         # dataframe

            "GoodnessofFit"     : gof,                      # float
            "Accuracy"          : B_fixed_accuracy,         # float
            "Quantity"          : B_denovo_quantity,        # bool
            "Quality"           : B_denovo_quality,         # float

            "Tprofile"          : Tprofile,                 # <class 'str'>
            "Iprofile"          : Iprofile                  # <class 'str'>
            }
    except:
        output = 0

    return output



