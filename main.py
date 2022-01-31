import SigPhylo
import csv
import utilities

# Questions?
# 1. non-negativity and normalization are done also inside the variational inference calculation
# 2. transfer coeff multiplied by pure alpha or preprocessed (non-negativity and normalizing)

'''
input = {
    "M_path" : "/home/azad/Documents/thesis/SigPhylo/data/simulated/data_sigphylo.csv",
    "beta_fixed_path" : "/home/azad/Documents/thesis/SigPhylo/data/simulated/beta_fixed.csv",
    "A_path" : "/home/azad/Documents/thesis/SigPhylo/data/simulated/A.csv",
    "k_denovo" : 1,

    "hyper_lambda" : 1,
    "lr" : 0.05,
    "steps_per_iter" : 500,
    "max_iter" : 100,
    "epsilon" : 0.0001
    }

L = SigPhylo.inference(input)
'''
utilities.generate_data()

def run_over_lambda(hyper_lambda):
    res = []
    for i in hyper_lambda:
        print("lambda =", i)

        input = {
            "M_path" : "/home/azad/Documents/thesis/SigPhylo/data/simulated/data_sigphylo.csv",
            "beta_fixed_path" : "/home/azad/Documents/thesis/SigPhylo/data/simulated/beta_fixed.csv",
            "A_path" : "/home/azad/Documents/thesis/SigPhylo/data/simulated/A.csv",
            "k_denovo" : 1,

            "hyper_lambda" : i,
            "lr" : 0.05,
            "steps_per_iter" : 500,
            "max_iter" : 100,
            "epsilon" : 0.0001
            }

        L = SigPhylo.inference(input)
        res.append(L)

    # likelihoods over lambdas
    with open("data/results/likelihoods.csv", 'w') as f:
        write = csv.writer(f)
        itr = len(hyper_lambda)
        for w in range(itr):
            write.writerow([hyper_lambda[w], res[w]])

hyper_lambda = [0, 0.2, 0.4, 0.6, 0.8, 1]
run_over_lambda(hyper_lambda)
