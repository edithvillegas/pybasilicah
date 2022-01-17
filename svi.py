import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import model
import guide
import model_beta_fixed



def single_inference(params):
    
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

    adam_params = {"lr": params["lr"]}
    optimizer = Adam(adam_params)
    elbo = Trace_ELBO()

    svi = SVI(model.model, guide.guide, optimizer, loss=elbo)

#   inference - do gradient steps
    for step in range(params["steps_per_iter"]):
        loss = svi.step(params)