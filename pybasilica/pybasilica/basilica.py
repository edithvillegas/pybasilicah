import torch
import numpy as np
import pandas as pd
from pybasilica.utilities import fixedFilter
from pybasilica.utilities import denovoFilter
from pybasilica.utilities import stopRun
from pybasilica.run import multi_k_run


def pyfit(M, B_input, k_list, cosmic_df, lr, steps_per_iter, fixedLimit, denovoLimit):
    # M ------------- dataframe
    # B_input ------- dataframe
    # k_list -------- list
    # cosmic_path --- dataframe

    theta = np.sum(M.values, axis=1)
    params = {
        "M" :               torch.tensor(M.values).float(), 
        "beta_fixed" :      torch.tensor(B_input.values).float(), 
        "lr" :              lr, 
        "steps_per_iter" :  steps_per_iter
        }

    counter = 1
    while True:
        #print("Loop", counter, "==================================")
        #print("Beta Input:", list(B_input.index))

        # k_list --- dtype: list
        k_inf, A_inf, B_inf = multi_k_run(params, k_list)
        # k_inf --- dtype: int
        # A_inf --- dtype: torch.Tensor
        # B_inf --- dtype: torch.Tensor

        # A_inf ----- dtype: torch.Tensor
        # B_input --- dtype: data.frame
        # theta ----- dtype: numpy
        B_input_sub = fixedFilter(A_inf, B_input, theta, fixedLimit)
        # B_input_sub ---- dtype: list
        #print("Alpha Inferred:\n", A_inf)
        #print("Beta Fixed sub:\n", B_input_sub)
        
        if k_inf > 0:
            # B_inf -------- dtype: torch.Tensor
            # cosmic_df ---- dtype: dataframe
            B_input_new = denovoFilter(B_inf, cosmic_df, denovoLimit)
            # B_input_new --- dtype: list
        else:
            B_input_new = []
        
        '''
        print("Beta input new:\n", B_input_new)
        print("          Input COSMIC Signature:", list(B_input.index))
        print("Detected Signatures (from Input):", B_input_sub)
        if B_inf != "NA":
            print("     Out of Input Signatures:", B_inf.size()[0])
        print("  New Detected COSMIC Signatures:", B_input_new)
        '''

        if stopRun(B_input_sub, list(B_input.index), B_input_new):
            signatures_inf = []
            for k in range(k_inf):
                signatures_inf.append("Unknown"+str(k+1))
            signatures = list(B_input.index) + signatures_inf
            mutation_features = list(B_input.columns)

            # alpha
            A_inf_np = np.array(A_inf)
            A_inf_df = pd.DataFrame(A_inf_np, columns=signatures)   # dataframe

            # beta
            if type(B_inf) is int:
            #if B_inf=="NA":
                B_inf_denovo_df = pd.DataFrame(columns=mutation_features)
            else:
                B_inf_denovo_np = np.array(B_inf)
                B_inf_denovo_df = pd.DataFrame(B_inf_denovo_np, index=signatures_inf, columns=mutation_features)
            #B_full = torch.cat((params["beta_fixed"], B_inf), axis=0)
            
            B_inf_fixed_df = B_input    # dataframe

            return A_inf_df, B_inf_fixed_df, B_inf_denovo_df
            # A_inf_df ---------- dtype: dataframe
            # B_inf_fixed_df ---- dtype: dataframe
            # B_inf_denovo_df --- dtype: dataframe

        
        B_input = cosmic_df.loc[B_input_sub + B_input_new]  # dtype: dataframe
        params["beta_fixed"] = torch.tensor(B_input.values).float()
        
        counter += 1

