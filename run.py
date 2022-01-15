import numpy as np
import pandas as pd
import torch
import infer
import utilities


input = {
    "M_path" : "/home/azad/Documents/thesis/SigPhylo/data/data_sigphylo.csv",
    "beta_fixed_path" : "/home/azad/Documents/thesis/SigPhylo/data/beta_aging.csv",
    "A_path" : "/home/azad/Documents/thesis/SigPhylo/data/A.csv",
    "k_denovo" : 1,

    "hyper_lambda" : 0.6,
    "lr" : 0.05,
    "steps_per_iter" : 500,
    "max_iter" : 100,
    "epsilon" : 0.001
}

infer.full_inference(input)