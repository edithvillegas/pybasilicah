import SigPhylo
import csv
import utilities
import os
import shutil
import json

utilities.generate_data()

def batch_run(k_list, lambda_list, folder_name):

    # create new directory (overwrite if exist)
    new_dir = "data/results/" + folder_name
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.mkdir(new_dir)

    Data = {}
    i = 1

    for k in k_list:
        for l in lambda_list:

            print("k_denovo =", k, "| lambda =", l)

            input = {
                "folder" : "new",
                "M_path" : "/home/azad/Documents/thesis/SigPhylo/data/simulated/data_sigphylo.csv",
                "beta_fixed_path" : "/home/azad/Documents/thesis/SigPhylo/data/simulated/beta_fixed.csv",
                "A_path" : "/home/azad/Documents/thesis/SigPhylo/data/simulated/A.csv",
                "k_denovo" : k,
                "hyper_lambda" : l,
                }

            data, Likelihood = SigPhylo.inference(input)
            #encodedNumpyData = json.dumps(data, cls=utilities.NumpyArrayEncoder)

            # likelihoods over lambdas
            with open("data/results/" + input["folder"] + "/likelihoods.csv", 'a') as f:
                write = csv.writer(f)
                write.writerow([k, l, Likelihood])

            Data[str(i)] = data
            i += 1

    with open("data/results/" + input["folder"] + "/data.json", 'w') as outfile:
        json.dump(Data, outfile, cls=utilities.NumpyArrayEncoder)


lambda_list = [0, 0.5, 1]
K_list = [1,2]
batch_run(K_list, lambda_list, "new")
