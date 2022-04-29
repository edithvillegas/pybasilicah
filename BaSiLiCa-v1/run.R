#===============================================================================
#============================== FULL RUN =======================================
#===============================================================================
library(reticulate)
library(tidyr)      # tibble()
library(dplyr)      # add_row()
library(ggplot2)    # ggplot()

#getwd() # mostly starts at "/home/azad"
#setwd("/home/azad")
#use_condaenv("SigPhylo")
#setwd("/home/azad/Documents/thesis/SigPhylo/PyBaSiLiCa")
#source_python("utilities.py")
#source_python("basilica.py")
#setwd("/home/azad/Documents/thesis/SigPhylo") # change the directory
#import("PyBaSiLiCa")
#setwd("/home/azad/Documents/thesis/SigPhylo/PyBaSiLiCa")
source_python("simulation.py")


M <- readCatalogue("/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv")
beta_input <- readBeta("/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv")
beta_cosmic <- readBeta("/home/azad/Documents/thesis/SigPhylo/data/cosmic/cosmic_catalogue.csv")
beta_expected <- readBeta("/home/azad/Documents/thesis/SigPhylo/data/real/expected_beta.csv")

x <- fitModel(M, beta_input, k_list, beta_cosmic, 0.05, 0.9)
alpha <- x[["Alpha"]]
beta_fixed <- x[["Beta_Fixed"]]
beta_denovo <- x[["Beta_Denovo"]]

plot_alpha(alpha)
plot_beta(beta_fixed, useRowNames = TRUE)
plot_beta(beta_denovo, useRowNames = TRUE)

a <- plot_beta(beta_denovo, useRowNames = TRUE)
b <- plot_beta(B_exp, useRowNames = TRUE)

grid.arrange(a, b)

beta_expected <- readBeta("/home/azad/Documents/thesis/SigPhylo/data/real/expected_beta.csv")
B_exp <- beta_expected["Signature 3(missing from COSMIC)",]





