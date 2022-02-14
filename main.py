import SigPhylo
import utilities

input = {
    "M_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv", 
    "beta_fixed_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv", 
    "A_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/A.csv", 
    "k_list" : [1, 2], 
    "lambda_list" : [0, 0.1], 
    "lr"                : 0.05, 
    "steps_per_iter"    : 500, 
    "max_iter"          : 5, 
    "epsilon"           : 0.0001, 
    "dir" : "/home/azad/Documents/thesis/SigPhylo/data/results/output2", 
    }

SigPhylo.batch_run(input)

#utilities.generate_data()
