


#===============================================================================
#======================== GENERATE PDF FILE ====================================
#===============================================================================

# load data --------------------------------------------------------------------
json_path = "/home/azad/Documents/thesis/SigPhylo/data/results/output5/output.json"
data <- load_data(json_path)

# output file ------------------------------------------------------------------
output_dir <- "/home/azad/Documents/thesis/SigPhylo/data/results/"

#-------------------------------------------------------------------------------
# export to pdf
#-------------------------------------------------------------------------------
pdf(file = paste(output_dir, "results.pdf", sep = ""))

#-------------------------------------------------------------------------------
# page 01 | overall insights ---> 
# k, lambda, log-likelihood, BIC & cosine similarity for constructed phylogeny
#-------------------------------------------------------------------------------
plot_k_loglike(data)
plot_k_bic(data)
plot_lambda_loglike(data)
plot_lambda_bic(data)
heatmap(data)
plot_cosine(data, 0.95)

# get best run info ------------------------------------------------------------
best_loglike_info <- best_loglike(data)
k_loglike <- best_loglike_info[["k"]]
lambda_loglike <- best_loglike_info[["lambda"]]

best_bic_info <- best_bic(data)
k_bic <- best_bic_info[["k"]]
lambda_bic <- best_bic_info[["lambda"]]

#-------------------------------------------------------------------------------
# page 02 | alpha ---> plot alpha for best BIC
#-------------------------------------------------------------------------------
alpha <- alpha_read_tibble(data, k_bic, lambda_bic) # it seems no need to this function
plot_alpha(data, k_bic)

#-------------------------------------------------------------------------------
# page 03 | beta fixed ---> plot beta_fixed
#-------------------------------------------------------------------------------
beta_fixed_path <- "/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv"
beta_fixed <- beta_read_csv(beta_fixed_path)
plot_beta(beta_fixed)

#-------------------------------------------------------------------------------
# page 04 | beta denovo ---> plot beta_denovo for best BIC
#-------------------------------------------------------------------------------
beta_denovo <- beta_read_tibble(data, k_bic, lambda_bic)
plot_beta(beta_denovo)

#-------------------------------------------------------------------------------
# THE END
#-------------------------------------------------------------------------------
dev.off()

#plot_list <- list(m_plot, fixed_plot)
#ggarrange(plotlist = plot_list, ncol = 1)
