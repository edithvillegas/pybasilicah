import torch
import pyro
import pyro.distributions as dist
import numpy as np
from pyro.infer.autoguide import AutoDelta
from pyro.infer.autoguide.initialization import init_to_sample
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt
import pandas as pd