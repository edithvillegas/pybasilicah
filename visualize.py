from itertools import groupby
from operator import index
from unicodedata import name
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import utilities
import joypy
import seaborn as sns
import json


# visualize M
def phylogeny(json_path):
    
    with open(json_path, "r") as f:
        data = json.load(f)

    M = data["input"]["M"]
    F = data["input"]["mutation_features"]
    df = pd.DataFrame(M, columns=F)
    n = df.shape[0]

    L = []
    for i in range(n):
        L.append("".join(["Branch", str(i+1)]))
    df.index = L
    
    fig = plt.figure(figsize=(15,10))
    for i, (name, row) in enumerate(df.iterrows()):
        # row: <class 'pandas.core.series.Series'>
        # name: <class 'str'>
        ax = plt.subplot(n,1, i+1)
        ax.set_title(row.name)
        ax.bar(row.index, row)

    fig.suptitle(" ".join(["Phylogeny with", str(n), "Branches"]))
    fig.tight_layout()
    plt.show()

#------------------------ DONE! ----------------------------------
def beta_csv(beta_path):
    df = pd.read_csv(beta_path, index_col=0)
    k = df.shape[0]

    fig, axes = plt.subplots(nrows=k, ncols=1)
    plt.suptitle("Infered Betas")
    for i, (name, row) in enumerate(df.iterrows()):
    #for i in range(k):
        #xpoints = np.array(range(96))
        #sub_df = df.iloc[i]
        #plt.bar(xpoints, ypoints)
        #sub_df.plot(kind="bar", stacked=False, ax=axes[i])
        row.plot(kind="bar", ax=axes[i])

    plt.show()

#=============================================================================================
#=============================================================================================
#=============================================================================================

def likelihood_all(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)["output"]
    df = pd.DataFrame(columns=["k_denovo", "lambda", "LH"])

    for key, value in data.items():
        k = value["k_denovo"]
        landa = value["lambda"]
        L = value["likelihoods"][-1]
        x_numpy = np.array([k, landa, L])
        x_series = pd.Series(x_numpy, index=["k_denovo", "lambda", "LH"])
        df = df.append(x_series, ignore_index=True)
    
    pd.pivot_table(
        df.reset_index(), 
        index='k_denovo', 
        columns='lambda', 
        values='LH'
        ).plot(subplots=True)
    plt.show()

#--------------------------------------------------------------------------------------------------

def likelihood_best_k(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)["output"]
    df = pd.DataFrame(columns=["k_denovo", "lambda", "LH"])
    k, l = utilities.best(json_path)


    for key, value in data.items():
        if (value["k_denovo"]==k):
            print("hello world!")

        k = value["k_denovo"]
        landa = value["lambda"]
        L = value["likelihoods"][-1]
        x_numpy = np.array([k, landa, L])
        x_series = pd.Series(x_numpy, index=["k_denovo", "lambda", "LH"])
        df = df.append(x_series, ignore_index=True)
    
    pd.pivot_table(
        df.reset_index(), 
        index='k_denovo', 
        columns='lambda', 
        values='LH'
        ).plot(subplots=True)
    plt.show()

#--------------------------------------------------------------------------------------------------

def exposure_best_k(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)["output"]

    df = pd.DataFrame(columns=["k_denovo", "lambda", "LH"])

    for key, value in data.items():
        k = value["k_denovo"]  # int
        landa = value["lambda"]    # int
        L = value["likelihoods"][-1]   # float
        x_numpy = np.array([k, landa, L])
        x_series = pd.Series(x_numpy, index=["k_denovo", "lambda", "LH"])
        df = df.append(x_series, ignore_index=True)

    k_max = df.iloc[df['LH'].idxmax()]["k_denovo"]
    n = df[df["k_denovo"]== k_max].shape[0]

    xx = {}
    for key, value in data.items():
        if value["k_denovo"]==k_max:
            xx[str(value["lambda"])] = value["alpha"]
            #print(value["k_denovo"], value["lambda"], value["alpha"])
            #x = pd.DataFrame(value["alpha"])

    L = []
    for key, value in xx.items():
        x_numpy = np.array(value)
        #x_series = pd.Series(x_numpy, index=["k_denovo", "lambda", "LH"])
        L.append(pd.DataFrame(x_numpy))

    #print(type(L[2]))
    fig, axes = plt.subplots(nrows=n, ncols=1)
    plt.suptitle("Signature Profile Distribution")
    #plt.title("")
    #plt.xlabel("Branches")
    #plt.ylabel("Exposure (Relative)")
    for i in range(len(L)):
        L[i].plot(kind="bar", stacked=True, ax=axes[i])

    plt.show()

#--------------------------------------------------------------------------------------------------


'''
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

def alpha_convergence(infered_alpha_path, expected_alpha_path, branch):
    expected_alpha = pd.read_csv(expected_alpha_path, header=None)
    #expected_alpha = torch.tensor(expected_alpha.values)             # dtype:torch.Tensor
    #expected_alpha = expected_alpha.float()

    infered_alpha = pd.read_csv(infered_alpha_path, header=None)
    #infered_alpha = torch.tensor(infered_alpha.values)             # dtype:torch.Tensor
    #infered_alpha = infered_alpha.float()

    print(expected_alpha)
    print(infered_alpha)

    n = expected_alpha.shape[0]             # no. of branches
    k = expected_alpha.shape[1]             # no. of signatures
    itr = int(infered_alpha.shape[0] / n)   # no. of iterations

    xpoints = np.array(range(itr))

    legends = []
    for l in range(k):
        legends.append("alpha" + str(l+1))

    # iterate over signature profile
    for i in range(k):
        r = expected_alpha.loc[branch][i]
        values = []

        for j in range(branch, infered_alpha.shape[0], n):
            
            if (r == 0.00):
                value = float("{:.3f}".format(1.0 - infered_alpha.iloc[j][i]))
                #print("r0", value)
            else:
                value = float("{:.3f}".format(infered_alpha.iloc[j][i] / r))
                #print("r1", value)
            values.append(value)
        
        ypoints = np.array(values)
        plt.plot(xpoints, ypoints, label=legends[i])

    plt.title("alpha convergence")
    plt.xlabel("iterations")
    plt.ylabel("alpha ratio")
    plt.grid()
    plt.legend()
    plt.show()


#for i in range(n):
#plt.subplot(1, n, i+1)
#plt.plot(xpoints, ypoints, label=legends[j]+" : real = "+str(format(target[branch][j].item(), '.2f'))+")")
print(
    "real :", r, 
    "infered :", infered_alpha.iloc[t][signature], 
    "ratio :", float("{:.3f}".format(infered_alpha.iloc[t][signature] / r))
    )   
'''