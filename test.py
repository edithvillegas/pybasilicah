import pandas as pd
import numpy as np
import torch
import pyro.distributions as dist
import pyro


my_path = "/home/azad/Documents/thesis/SigPhylo/data/"
data_file = "data_sigphylo.csv"
aging_file = "beta_aging.csv"
expected_beta = "expected_beta.csv"

# load data
M = pd.read_csv(my_path + data_file)
beta_aging = pd.read_csv(my_path + aging_file)
expected_beta = pd.read_csv(my_path + expected_beta)


contexts = list(beta_aging.columns[1:])
signature_names = list(beta_aging.values[:, 0])
counts = beta_aging.values[:,1:]
counts = torch.tensor(np.array(counts, dtype=float))
counts = counts.float()

#print(contexts)
#print(alpha)

a = torch.tensor([[1,2,3,4,5],
[6,7,8,9,10],
[11,12,13,14,15]])

'''
with pyro.plate("K", 3):
        with pyro.plate("N", 5):
            alpha = pyro.sample("activities", dist.Normal(a, 1))

'''
a = pyro.sample("activities", dist.Normal(a, 1))
print(a)