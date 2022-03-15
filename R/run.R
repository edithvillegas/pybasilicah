#===============================================================================
#============================== FULL RUN =======================================
#===============================================================================

library(reticulate)
getwd() # mostly starts at "/home/azad"
setwd("/home/azad/Documents/thesis/SigPhylo/python") # change the directory

#============================== CREATE INPUT ===================================

a <- dict(
  k_list            = c(1, 2), 
  lambda_list       = c(0.0, 0.1),
  dir               = "/home/azad/Documents/thesis/SigPhylo/data/results/test",
  sim               = FALSE, 
  parallel          = FALSE, 
  
  M_path            = "/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv", 
  beta_fixed_path   = "/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv", 
  A_path            = "/home/azad/Documents/thesis/SigPhylo/data/real/A.csv", 
  expected_beta_path= "/home/azad/Documents/thesis/SigPhylo/data/real/expected_beta_2.csv",
  
  cosmic_path       = "/home/azad/Documents/thesis/SigPhylo/cosmic/cosmic_catalogue.csv"
)

#======================== RUN PYTHON SCRIPTS ===================================
source_python("batch.py")
batch_run(b)

#-------------------------------------------------------------------------------
# plot phylogeny
#-------------------------------------------------------------------------------
library(ggplot2)
library(ggtree)
library(dplyr)

# ggtree BRMA object and plot
brma = ggtree::read.tree("/home/azad/Documents/thesis/simple.nwk")

brma_plot = ggtree(brma, ladderize=F) + 
  #geom_tiplab() + 
  ggtitle("BRMA (sample tree)") + 
  theme_tree() + 
  #geom_nodepoint(aes(color=isTip), color = 'purple4', size=3) + 
  geom_tippoint()
  #geom_cladelabel('B2d', 'jj') + 
  #geom_tiplab() + 
  #geom_text(aes(label=label), size=3, color="purple", hjust=-.3) + 
  #geom_treescale(fontsize=3, linesize=1, offset=.2, x = 0) + 
  #coord_cartesian(clip = 'off') + 
  #xlim(0, 9000)

brma_plot + geom_text(aes(label=node))



