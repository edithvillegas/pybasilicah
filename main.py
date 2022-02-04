import SigPhylo
import csv
import utilities
import os
import shutil


utilities.generate_data()

def batch_run(k_list, lambda_list, folder_name):

    # create new directory (overwrite if exist)
    new_dir = "data/results/" + folder_name
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.mkdir(new_dir)

    for k in k_list:
        for l in lambda_list:

            print("k_denovo =", k, "| lambda =", l)

            input = {
                "folder" : folder_name,
                "M_path" : "/home/azad/Documents/thesis/SigPhylo/data/simulated/data_sigphylo.csv",
                "beta_fixed_path" : "/home/azad/Documents/thesis/SigPhylo/data/simulated/beta_fixed.csv",
                "A_path" : "/home/azad/Documents/thesis/SigPhylo/data/simulated/A.csv",
                "k_denovo" : k,
                "hyper_lambda" : l,
                }

            # likelihoods over lambdas
            with open("data/results/" + input["folder"] + "/likelihoods.csv", 'a') as f:
                write = csv.writer(f)
                write.writerow([k, l, SigPhylo.inference(input)])



lambda_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
K_list = [1]
batch_run(K_list, lambda_list, "KL")
