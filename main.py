import SigPhylo
import csv
import utilities
import os
import shutil
import json
import numpy as np

utilities.generate_data()

def batch_run(k_list, lambda_list, folder_name):

    # create new directory (overwrite if exist)
    new_dir = "data/results/" + folder_name
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.mkdir(new_dir)

    Data = {}
    likelihoods = []
    i = 1

    for k in k_list:
        for landa in lambda_list:

            print("k_denovo =", k, "| lambda =", landa)

            input = {
                "folder" : new_dir,
                "M_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv",
                "beta_fixed_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv",
                "beta_fixed_name" : ["SBS5"], 
                "A_path" : "/home/azad/Documents/thesis/SigPhylo/data/real/A.csv",
                "k_denovo" : k,
                "hyper_lambda" : landa,
                }

            data, L = SigPhylo.inference(input)
            likelihoods.append(L)
            #encodedNumpyData = json.dumps(data, cls=utilities.NumpyArrayEncoder)

            # likelihoods over lambdas
            with open(input["folder"] + "/likelihoods.csv", 'a') as f:
                write = csv.writer(f)
                write.writerow([k, landa, L])

            Data[str(i)] = data
            i += 1

    ignore1, xx = utilities.M_read_csv(input["M_path"])
    ignore2, ignore3, yy = utilities.beta_read_csv(input["beta_fixed_path"])

    Data["M"] = np.array(xx)
    Data["beta_fixed"] = np.array(yy)
    Data["likelihoods"] = likelihoods

    with open(input["folder"] + "/data.json", 'w') as outfile:
        json.dump(Data, outfile, cls=utilities.NumpyArrayEncoder)


lambda_list = [0, 1]
K_list = [1]
batch_run(K_list, lambda_list, "new")
