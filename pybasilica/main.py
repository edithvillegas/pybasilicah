import torch
import numpy as np
import pandas as pd

#from pybasilica.utilities import fixedFilter
#from pybasilica.utilities import denovoFilter
#from pybasilica.utilities import stopRun
#from pybasilica.run import multi_k_run
from utilities import fixedFilter
from utilities import denovoFilter
from utilities import stopRun
from utilities import initialize_params
from run import multi_k_run




def pyfit(M, groups, B_input, cosmic_df, k, lr, steps, phi, delta):
    # M ------------- dataframe
    # B_input ------- dataframe
    # k ------------- list
    # cosmic_path --- dataframe

    theta = np.sum(M.values, axis=1)

    params = initialize_params(M, groups, B_input, lr, steps)

    counter = 1
    while True:

        # k_list --- dtype: list
        k_inf, A_inf, B_inf = multi_k_run(params, k)
        # k_inf --- dtype: int
        # A_inf --- dtype: torch.Tensor
        # B_inf --- dtype: torch.Tensor

        # A_inf ----- dtype: torch.Tensor
        # B_input --- dtype: data.frame
        # theta ----- dtype: numpy
        B_input_sub = fixedFilter(A_inf, B_input, theta, phi)
        # B_input_sub ---- dtype: list
        
        # B_inf -------- dtype: torch.Tensor
        # cosmic_df ---- dtype: dataframe
        B_input_new = denovoFilter(B_inf, cosmic_df, delta)
        # B_input_new --- dtype: list

        if B_input is None:
            B_input_list = []
        else:
            B_input_list = list(B_input.index)

        if stopRun(B_input_sub, B_input_list, B_input_new):
            signatures_inf = []
            for k in range(k_inf):
                signatures_inf.append("Unknown"+str(k+1))
            signatures = B_input_list + signatures_inf
            mutation_features = list(M.columns)

            # alpha
            A_inf_np = np.array(A_inf)
            A_inf_df = pd.DataFrame(A_inf_np, columns=signatures)   # dataframe

            # beta
            if B_inf is None:
                B_inf_denovo_df = pd.DataFrame(columns=mutation_features)
            else:
                B_inf_denovo_np = np.array(B_inf)
                B_inf_denovo_df = pd.DataFrame(B_inf_denovo_np, index=signatures_inf, columns=mutation_features)
            
            if B_input is None:
                B_inf_fixed_df = pd.DataFrame(columns=mutation_features)
            else:
                B_inf_fixed_df = B_input    # dataframe

            return A_inf_df, B_inf_fixed_df, B_inf_denovo_df
            # A_inf_df ---------- dtype: dataframe
            # B_inf_fixed_df ---- dtype: dataframe
            # B_inf_denovo_df --- dtype: dataframe

        
        B_input = cosmic_df.loc[B_input_sub + B_input_new]  # dtype: dataframe
        params["beta_fixed"] = torch.tensor(B_input.values).float()
        
        counter += 1

