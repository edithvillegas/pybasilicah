import batch
import torch
import utilities


#--------------------------------------------------------------------------------
# Model Input Data
#--------------------------------------------------------------------------------

alpha = torch.tensor(
    [[0.95, 0.05], 
    [0.40, 0.60], 
    [0.04, 0.96]]
    )
fixed_signatures = ["SBS5"]
denovo_signatures = ["SBS84"]
theta = [1200, 3600, 2300]
A = torch.tensor(
    [[1,1,1], 
    [1,1,1], 
    [1,1,1]])

arg_list = {
    "k_list"            : [1, 2], 
    "lambda_list"       : [0, 0.1], 
    "dir"               : "/home/azad/Documents/thesis/SigPhylo/data/results/test", 
    "sim_mode"          : False, 
    "parallel_mode"     : False, 

    "real_data" : {
        "M_path"            : "/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv", 
        "beta_fixed_path"   : "/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv", 
        "A_path"            : "/home/azad/Documents/thesis/SigPhylo/data/real/A.csv", 
        "expected_beta_path": "/home/azad/Documents/thesis/SigPhylo/data/real/expected_beta_2.csv"
        }, 
    
    "sim_data" : {
        "alpha"             : alpha, 
        "fixed_signatures"  : fixed_signatures, 
        "denovo_signatures" : denovo_signatures, 
        "theta"             : theta, 
        "A_tensor"          : A
    }, 

    "cosmic_path"           : "/home/azad/Documents/thesis/SigPhylo/cosmic/cosmic_catalogue.csv"
    }

#--------------------------------------------------------------------------------
# Run Model
#--------------------------------------------------------------------------------
if __name__ == "__main__":
    batch.batch_run(arg_list)


