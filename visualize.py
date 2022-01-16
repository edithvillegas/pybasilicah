import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import utilities
import joypy
import seaborn as sns


def catalogue(path):
    M_df = pd.read_csv(path)    # dtype: DataFrame

    mutation_features = list(M_df.columns)

    M_np = M_df.values          # dtype: numpy.ndarray
    M = torch.tensor(M_np)      # dtype: torch.Tensor
    M = M.float()               # dtype: torch.Tensor
    
    n = M.size()[0]
    for i in range(n):
        label= ("branch " + str(i+1))
        xpoints = range(0, 96)
        ypoints = np.array(M.iloc[i])
        plt.subplot(n, 1, i+1)
        plt.bar(xpoints, ypoints, label=label, color="g")

        plt.suptitle("genome catalogue")

        #plt.title("genome catalogue")
        plt.xlabel("mutation features")
        plt.ylabel("number of mutations")

        plt.grid()
        plt.legend()
    plt.show()

def alpha_convergence(infered_alpha, expected_alpha, branch):

    n = expected_alpha.shape[0]             # no. of branches
    k = expected_alpha.shape[1]             # no. of signatures
    itr = int(infered_alpha.shape[0] / n)   # no. of iterations

    xpoints = np.array(range(itr))

    legends = []
    for l in range(k):
        legends.append("alpha" + str(l+1))

    # iterate over signature profile
    for i in range(k):
        r = expected_alpha[branch][i]
        values = []

        for j in range(branch, infered_alpha.shape[0], n):
            value = float("{:.3f}".format(infered_alpha.iloc[j][i] / r))
            values.append(value)
        
        ypoints = np.array(values)
        plt.plot(xpoints, ypoints, label=legends[i])

    plt.title("alpha convergence")
    plt.xlabel("iterations")
    plt.ylabel("alpha ratio")
    plt.grid()
    plt.legend()
    plt.show()


# not completed
def beta(b):

    k = b.shape[0]  # no. of signatures

    xpoints = np.array(range(96))

    for i in range(k):
        ypoints = np.array(b.iloc[i])
        plt.bar(xpoints, ypoints)
    
    plt.show()

def priors():
    joypy.joyplot()

'''
#for i in range(n):
#plt.subplot(1, n, i+1)
#plt.plot(xpoints, ypoints, label=legends[j]+" : real = "+str(format(target[branch][j].item(), '.2f'))+")")
print(
    "real :", r, 
    "infered :", infered_alpha.iloc[t][signature], 
    "ratio :", float("{:.3f}".format(infered_alpha.iloc[t][signature] / r))
    )   
'''