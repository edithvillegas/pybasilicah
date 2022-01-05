import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import aux


def vis_alpha(alphas, branch):

    print("visualizing alphas in branch", branch)

    Ralpha = torch.tensor([
        [0.35, 0.50, 0.15],
        [0.52, 0.43, 0.05],
        [0.51, 0.45, 0.04],
        [0.02, 0.03, 0.95],
        [0.23, 0.46, 0.31]
        ])

    n = Ralpha.size()[0]
    m = Ralpha.size()[1]

    xpoints = np.array(range(len(alphas)))
    legends = ["alpha1", "alpha2", "alpha3"]

    #for i in range(n):
        #plt.subplot(1, n, i+1)

    
    # branch number
    i=3

    for j in range(m):

        r = Ralpha[i][j].item()

        vals = []
        for k in range(len(alphas)):
            c = alphas[k][i][j].item()
            vals.append(c)

        ypoints = np.array(vals)
        
        plt.plot(xpoints, ypoints, label=legends[j]+" : real = "+str(format(Ralpha[i][j].item(), '.2f'))+")")


    plt.title("alpha change over iterations")
    plt.xlabel("iterations")
    plt.ylabel("alpha value")
    plt.grid()
    plt.legend()
    plt.show()


# not completed
def vis_beta():
    my_path = "/home/azad/Documents/thesis/SigPhylo/data/"
    beta_file = "expected_beta.csv"
    # load data
    beta_full = pd.read_csv(my_path + beta_file)
    beta, signature_names, contexts = aux.get_signature_profile(beta_full)
