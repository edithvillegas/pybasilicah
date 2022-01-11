import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import aux


# for first branch
def alpha(infered, target):

    n = target.size()[0]  # no. of branches
    k = target.size()[1]  # no. of signatures
    itr = len(target)     # no. of iterations

    xpoints = np.array(range(itr))
    legends = []
    #legends = ["alpha1", "alpha2", "alpha3"]

    #for i in range(n):
        #plt.subplot(1, n, i+1)

    # branch number
    branch=0

    for j in range(k):

        legends.append("alpha" + str(j+1))
        r = target[branch][j].item()

        vals = []
        for t in range(len(infered)):
            c = infered[t][branch][j].item()
            vals.append(c)

        p = [x / r for x in vals]
        ypoints = np.array(p)
        
        #plt.plot(xpoints, ypoints, label=legends[j]+" : real = "+str(format(target[branch][j].item(), '.2f'))+")")
        plt.plot(xpoints, ypoints, label=legends[j])



    plt.title("alpha change over iterations")
    plt.xlabel("iterations")
    plt.ylabel("alpha value")
    plt.grid()
    plt.legend()
    plt.show()


# not completed
def beta(b):
    my_path = "/home/azad/Documents/thesis/SigPhylo/data/"
    beta_file = "expected_beta.csv"
    # load data
    beta_full = pd.read_csv(my_path + beta_file)
    beta, signature_names, contexts = aux.get_signature_profile(beta_full)
    k = b.size()[0]
    xpoints = np.array(range(96))
    ypoints = b[0]
    plt.plot(xpoints, ypoints)


def catalogue(m):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    #ax.set_xlim(0, 96)
    x = range(0, 96)
    y = m[0]
    #ax.bar(x, y)
    plt.bar(x, y, width= 0.9, align='center',color='cyan')

    ax.set_title("genome catalogue")
    ax.set_xlabel('mutation features')
    ax.set_ylabel('number of mutations')
    ax.set_xticks([0, 15, 31, 47, 63, 79, 95])
    ax.set_xticklabels([1, 16, 32, 48, 64, 80, 96])
    #plt.legend(labels = ['Total mutations'])
    plt.show()
