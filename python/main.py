import batch

# different systems check out:
# utilities ---> beta_read_name() ---> cosmic path
# batch ---> batch_run() ---> if sim==False ---> all paths

arg_list = {
    "k_list"        : [1, 2, 3, 4], 
    "lambda_list"   : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "dir"           : "/home/azad/Documents/thesis/SigPhylo/data/results/new",
    "sim"           : False
    }


batch.batch_run(arg_list)

