import batch


arg_list = {
    "k_list"        : [1, 2], 
    "lambda_list"   : [0, 0.1],
    "dir"           : "/home/azad/Documents/thesis/SigPhylo/data/results/rtest",
    "sim"           : False
    }


batch.batch_run(arg_list)

