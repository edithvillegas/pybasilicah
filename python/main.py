import SigPhylo

input = {
    "M_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv", 
    "beta_fixed_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv", 
    "A_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/A.csv", 

    "expected_beta_path" : "/home/azad/Documents/thesis/SigPhylo/data/simulated/beta_denovo.csv", 
    "expected_alpha_path" : "/home/azad/Documents/thesis/SigPhylo/data/simulated/expected_alpha.csv", 

    "k_list" : [1, 2], 
    "lambda_list" : [0, 0.1, 0.2, 0.3], 

    "lr"                : 0.05, 
    "steps_per_iter"    : 500, 
    "max_iter"          : 50, 
    "epsilon"           : 0.001, 
    "dir" : "/home/azad/Documents/thesis/SigPhylo/data/results/output44"
    }

SigPhylo.batch_run(input, sim=True)

#utilities.generate_data()
