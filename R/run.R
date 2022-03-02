#===============================================================================
#============================== FULL RUN =======================================
#===============================================================================

library(reticulate)
getwd() # mostly atarts at "/home/azad"
setwd("/home/azad/Documents/thesis/SigPhylo/python") # change the directory

#============================== CREATE INPUT ===================================
a = list(
  k_list = c(1, 2), 
  lambda_list = c(0, 0.1), 
  dir = "/home/azad/Documents/thesis/SigPhylo/data/results/test", 
  sim = FALSE, 
  M_path = "/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv", 
  beta_fixed_path = "/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv", 
  A_path = "/home/azad/Documents/thesis/SigPhylo/data/real/A.csv", 
  expected_beta_path = "/home/azad/Documents/thesis/SigPhylo/data/real/expected_beta_2.csv"
)
arg_list <- r_to_py(a)

#======================== RUN PYTHON SCRIPTS ===================================
source_python("main.py")
mainRun(arg_list)




