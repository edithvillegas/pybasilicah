import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import logging

import model
import guide



def single_inference(M, params, lr=0.05, num_steps=200):
    
    '''
    ====== inputs ======
    * M --> dtype:torch.Tensor
    * params = {"alpha" : alpha, 
                "beta" : beta,
                "k_denovo" : k_denovo, 
                "beta_fixed" : beta_counts, 
                "A" : A, 
                "lambda": 0.9}
    '''
    
    pyro.clear_param_store()  # always clear the store before the inference

    # learning global parameters

    adam_params = {"lr": lr}
    optimizer = Adam(adam_params)
    elbo = Trace_ELBO()

    svi = SVI(model.model, guide.guide, optimizer, loss=elbo)

    losses = []
#   inference - do gradient steps
    for step in range(num_steps):
        loss = svi.step(M, params)
        losses.append(loss)
        if step % 10 == 0:
            logging.info("Elbo loss: {}".format(loss))
