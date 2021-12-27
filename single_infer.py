import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import model

def single_inference(M, params, lr=0.05, num_steps=200):
    
    pyro.clear_param_store()  # always clear the store before the inference

    # learning global parameters

    adam_params = {"lr": lr}
    optimizer = Adam(adam_params)
    elbo = Trace_ELBO()

    svi = SVI(model.model, model.guide, optimizer, loss=elbo)

#   inference
#   do gradient steps
    for step in range(num_steps):
        loss = svi.step(M, params)
