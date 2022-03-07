import batch

arg_list = {
    "k_list"        : [1, 2], 
    "lambda_list"   : [0, 0.1, 0.2],
    "dir"           : "/home/azad/Documents/thesis/SigPhylo/data/results/test",
    "sim"           : False, 
    "parallel"      : True, 

    "M_path"        : "/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv", 
    "beta_fixed_path": "/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv", 
    "A_path"        : "/home/azad/Documents/thesis/SigPhylo/data/real/A.csv", 
    "expected_beta_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/expected_beta_2.csv",

    "cosmic_path" : "/home/azad/Documents/thesis/SigPhylo/cosmic/cosmic_catalogue.csv"
    }
    

if __name__ == "__main__":
    batch.batch_run(arg_list)


