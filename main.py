import SigPhylo
import utilities

input = {
    "M_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv", 
    "beta_fixed_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv", 
    "A_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/A.csv", 
    "K_list" : [1, 2, 3], 
    "lambda_list" : [0, 0.5, 1], 
    "dir" : "/home/azad/Documents/thesis/SigPhylo/data/results/output"
}

SigPhylo.batch_run(input)

#utilities.generate_data()
#beta_fixed_name = ["SBS5"]