import numpy as np
import pandas as pd
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.distributions as dist
import torch.nn.functional as F



class PyBasilica():

    def __init__(self, x, k_denovo, lr, n_steps, groups=None, beta_fixed=None):
        self._set_data_catalogue(x)
        self._set_beta_fixed(beta_fixed)
        self.k_denovo = int(k_denovo)
        self.lr = lr
        self.n_steps = int(n_steps)
        self._set_groups(groups)
        self._check_args()
    

    def _set_data_catalogue(self, x):
        try:
            self.x = torch.tensor(x.values).float()
            self.n_samples = x.shape[0]
        except:
            raise Exception("Invalid mutations catalogue, expected Dataframe!")
    
        
    def _set_beta_fixed(self, beta_fixed):
        try:
            self.beta_fixed = torch.tensor(beta_fixed.values).float()
            self.k_fixed = beta_fixed.shape[0]
        except:
            if beta_fixed is None:
                self.beta_fixed = None
                self.k_fixed = 0
            else:
                raise Exception("Invalid fixed signatures catalogue, expected DataFrame!")
    

    def _set_groups(self, groups):
        if groups is None:
            self.groups = None
        else:
            if isinstance(groups, list) and len(groups)==self.n_samples:
                self.groups = groups
            else:
                raise Exception("invalid groups argument, expected 'None' or a list with {} elements!".format(self.n_samples))


    def _check_args(self):
        if self.k_denovo==0 and self.k_fixed==0:
            raise Exception("No. of denovo and fixed signatures could NOT be zero at the same time!")
    
    
    def model(self):

        n_samples = self.n_samples
        k_fixed = self.k_fixed
        k_denovo = self.k_denovo
        groups = self.groups

        #----------------------------- [ALPHA] -------------------------------------
        if groups != None:

            #num_groups = max(params["groups"]) + 1
            n_groups = len(set(groups))
            alpha_tissues = dist.Normal(torch.zeros(n_groups, k_fixed + k_denovo), 1).sample()

            # sample from the alpha prior
            with pyro.plate("k", k_fixed + k_denovo):   # columns
                with pyro.plate("n", n_samples):        # rows
                    alpha = pyro.sample("latent_exposure", dist.Normal(alpha_tissues[groups, :], 1))
        else:
            alpha_mean = dist.Normal(torch.zeros(n_samples, k_fixed + k_denovo), 1).sample()

            with pyro.plate("k", k_fixed + k_denovo):   # columns
                with pyro.plate("n", n_samples):        # rows
                    alpha = pyro.sample("latent_exposure", dist.Normal(alpha_mean, 1))
        
        alpha = torch.exp(alpha)                                # enforce non negativity
        alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))     # normalize

        #----------------------------- [BETA] -------------------------------------
        if k_denovo==0:
            beta_denovo = None
        else:
            beta_mean = dist.Normal(torch.zeros(k_denovo, 96), 1).sample()
            with pyro.plate("contexts", 96):            # columns
                with pyro.plate("k_denovo", k_denovo):  # rows
                    beta_denovo = pyro.sample("latent_signatures", dist.Normal(beta_mean, 1))
            beta_denovo = torch.exp(beta_denovo)                                    # enforce non negativity
            beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))   # normalize

        #----------------------------- [LIKELIHOOD] -------------------------------------
        if self.beta_fixed is None:
            beta = beta_denovo
        elif beta_denovo is None:
            beta = self.beta_fixed
        else:
            beta = torch.cat((self.beta_fixed, beta_denovo), axis=0)
        
        with pyro.plate("contexts2", 96):
            with pyro.plate("n2", n_samples):
                pyro.sample("obs", dist.Poisson(torch.matmul(torch.matmul(torch.diag(torch.sum(self.x, axis=1)), alpha), beta)), obs=self.x)
                
    

    def guide(self):

        n_samples = self.n_samples
        k_fixed = self.k_fixed
        k_denovo = self.k_denovo
        #groups = self.groups

        alpha_mean = dist.Normal(torch.zeros(n_samples, k_fixed + k_denovo), 1).sample()

        with pyro.plate("k", k_fixed + k_denovo):
            with pyro.plate("n", n_samples):
                alpha = pyro.param("alpha", alpha_mean)
                pyro.sample("latent_exposure", dist.Delta(alpha))

        if k_denovo != 0:
            beta_mean = dist.Normal(torch.zeros(k_denovo, 96), 1).sample()
            with pyro.plate("contexts", 96):
                with pyro.plate("k_denovo", k_denovo):
                    beta = pyro.param("beta_denovo", beta_mean)
                    pyro.sample("latent_signatures", dist.Delta(beta))

    '''
    def _regularizer(self, beta_fixed, beta_denovo):

        if beta_denovo == None:
            dd = 0
        else:
            dd = 0
            c1 = 0
            for denovo1 in beta_denovo:
                c1 += 1
                c2 = 0
                for denovo2 in beta_denovo:
                    c2 += 1
                    if c1!=c2:
                        dd += F.kl_div(denovo1, denovo2, reduction="batchmean").item()

        if beta_fixed == None or beta_denovo == None:
            loss = 0
        else:
            #cosi = torch.nn.CosineSimilarity(dim=0)
            loss = 0
            for fixed in beta_fixed:
                for denovo in beta_denovo:
                    loss += F.kl_div(fixed, denovo, reduction="batchmean").item()
                    #loss += cosi(fixed, denovo).item()
            #print("loss:", loss)
        return loss + (dd/2)
    '''
    
    def _likelihood(self, M, alpha, beta_fixed, beta_denovo):
        
        if beta_fixed is None:
            beta = beta_denovo
        elif beta_denovo is None:
            beta = beta_fixed
        else:
            beta = torch.cat((beta_fixed, beta_denovo), axis=0)

        _log_like_matrix = dist.Poisson(torch.matmul(torch.matmul(torch.diag(torch.sum(M, axis=1)), alpha), beta)).log_prob(M)
        _log_like_sum = torch.sum(_log_like_matrix)
        _log_like = float("{:.3f}".format(_log_like_sum.item()))
        #print("loglike:",_log_like)

        return _log_like
    
    
    
    def _fit(self):
        
        pyro.clear_param_store()  # always clear the store before the inference

        # learning global parameters
        adam_params = {"lr": self.lr}
        optimizer = Adam(adam_params)
        elbo = Trace_ELBO()

        svi = SVI(self.model, self.guide, optimizer, loss=elbo)

        losses = []
        for step in range(self.n_steps):   # inference - do gradient steps
            loss = svi.step()
            losses.append(loss)
        
        self.losses = losses
        self._set_alpha()
        self._set_beta_denovo()
        self._set_bic()
        #self.likelihood = self._likelihood(self.x, self.alpha, self.beta_fixed, self.beta_denovo)
        #self.regularization = self._regularizer(self.beta_fixed, self.beta_denovo)



    def _set_alpha(self):
        # exposure matrix
        alpha = pyro.param("alpha").clone().detach()
        alpha = torch.exp(alpha)
        self.alpha = alpha / (torch.sum(alpha, 1).unsqueeze(-1))

    
    def _set_beta_denovo(self):
        # signature matrix
        if self.k_denovo == 0:
            self.beta_denovo = None
        else:
            beta_denovo = pyro.param("beta_denovo").clone().detach()
            beta_denovo = torch.exp(beta_denovo)
            self.beta_denovo = beta_denovo / (torch.sum(beta_denovo, 1).unsqueeze(-1))
    

    def _set_bic(self):

        M = self.x
        alpha = self.alpha

        _log_like = self._likelihood(M, alpha, self.beta_fixed, self.beta_denovo)

        k = (alpha.shape[0] * (alpha.shape[1])) + ((self.k_denovo + self.k_fixed) * M.shape[1])
        n = M.shape[0] * M.shape[1]
        bic = k * torch.log(torch.tensor(n)) - (2 * _log_like)

        self.bic = bic.item()


    
    def _convert_to_dataframe(self, x, beta_fixed):

        # mutations catalogue
        self.x = x
        sample_names = list(x.index)
        mutation_features = list(x.columns)

        # fixed signatures
        fixed_names = []
        if self.beta_fixed is not None:
            fixed_names = list(beta_fixed.index)
            self.beta_fixed = beta_fixed

        # denovo signatures
        denovo_names = []
        if self.beta_denovo is not None:
            for d in range(self.k_denovo):
                denovo_names.append("D"+str(d+1))
            self.beta_denovo = pd.DataFrame(np.array(self.beta_denovo), index=denovo_names, columns=mutation_features)

        # alpha
        self.alpha = pd.DataFrame(np.array(self.alpha), index=sample_names , columns= fixed_names + denovo_names)



