import SigPhylo
import utilities

input = {
    "M_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv", 
    "beta_fixed_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv", 
    "A_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/A.csv", 
    "k_list" : [1, 2], 
    "lambda_list" : [0, 0.1, 0.2, 0.3], 
    "lr"                : 0.05, 
    "steps_per_iter"    : 500, 
    "max_iter"          : 50, 
    "epsilon"           : 0.001, 
    "dir" : "/home/azad/Documents/thesis/SigPhylo/data/results/output5", 
    }

SigPhylo.batch_run(input)

#utilities.generate_data()
