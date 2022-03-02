import batch

# different systems check out:
# utilities ---> beta_read_name() ---> cosmic path
# batch ---> batch_run() ---> if sim==False ---> all paths

arg_list = {
    "k_list"        : [1, 2], 
    "lambda_list"   : [0, 0.1],
    "dir"           : "/home/azad/Documents/thesis/SigPhylo/data/results/test",
    "sim"           : False, 

    "M_path"        : "/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv", 
    "beta_fixed_path": "/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv", 
    "A_path"        : "/home/azad/Documents/thesis/SigPhylo/data/real/A.csv", 
    "expected_beta_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/expected_beta_2.csv"
    }

def mainRun(arg_list):
    batch.batch_run(arg_list)