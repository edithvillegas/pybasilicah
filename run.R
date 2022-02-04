#library(reticulate)

res_dir <- "/home/azad/Documents/thesis/SigPhylo/data/results/KL"
exp_dir <- "/home/azad/Documents/thesis/SigPhylo/data/simulated/"


#-------------------------------------------------------------------------------
# export to pdf
#-------------------------------------------------------------------------------
pdf(file = paste(res_dir, "/myPlot.pdf", sep = ""))

#-------------------------------------------------------------------------------
# page 01 : phylogeny & beta fixed
#-------------------------------------------------------------------------------
M <- paste(exp_dir, "data_sigphylo.csv", sep = "")
m_plot <- Phylogeny(M)

b_fixed <- paste(exp_dir, "beta_fixed.csv", sep = "")
fixed_plot <- Beta(b_fixed, "Fixed Beta")

plot_list <- list(m_plot, fixed_plot)
ggarrange(plotlist = plot_list, ncol = 1)

#-------------------------------------------------------------------------------
# page 02 : signatures shares
#-------------------------------------------------------------------------------
exp <- signature_share(paste(exp_dir, "expected_alpha.csv", sep = ""), "Expected")
L00 <- signature_share(paste(res_dir, "/lambda_0/alpha.csv", sep = ""), "Lambda=0")
L02 <- signature_share(paste(res_dir, "/lambda_0.2/alpha.csv", sep = ""), "Lambda=0.2")
L04 <- signature_share(paste(res_dir, "/lambda_0.4/alpha.csv", sep = ""), "Lambda=0.4")
L06 <- signature_share(paste(res_dir, "/lambda_0.6/alpha.csv", sep = ""), "Lambda=0.6")
L08 <- signature_share(paste(res_dir, "/lambda_0.8/alpha.csv", sep = ""), "Lambda=0.8")
L10 <- signature_share(paste(res_dir, "/lambda_1/alpha.csv", sep = ""), "Lambda=1")

plot_list <- list(L00, L02, L04, L06, L08, L10)
x <- ggarrange(plotlist = plot_list)
ggarrange(exp, x, ncol = 1)

#-------------TEST-------------------
exp <- signature_share(paste(exp_dir, "expected_alpha.csv", sep = ""), "Expected")

lambda_list <- c("Lambda=0", "Lambda=0.2", "Lambda=0.4", "Lambda=0.6", "Lambda=0.8", "Lambda=1")
j <- 1
plot_list = list()
files <- list.files(res_dir, pattern = "K_*", full.names = TRUE)
for (currentFile in files) {
  plot_list[[length(plot_list)+1]] <- signature_share(paste(currentFile, "/alpha.csv", sep = ""), lambda_list[j])
  #print(paste(currentFile, "/alpha.csv", sep = ""))
  j <- j + 1
}
x <- ggarrange(plotlist = plot_list)
ggarrange(exp, x, ncol = 1)
#-------------TEST-------------------

#-------------------------------------------------------------------------------
# page 03 : beta (denovo vs inferred)
#-------------------------------------------------------------------------------
a00 <- Beta(paste(res_dir, "/lambda_0/beta.csv", sep = ""), "lambda 0")
a02 <- Beta(paste(res_dir, "/lambda_0.2/beta.csv", sep = ""), "lambda 0.2")
a04 <- Beta(paste(res_dir, "/lambda_0.4/beta.csv", sep = ""), "lambda 0.4")
a06 <- Beta(paste(res_dir, "/lambda_0.6/beta.csv", sep = ""), "lambda 0.6")
a08 <- Beta(paste(res_dir, "/lambda_0.8/beta.csv", sep = ""), "lambda 0.8")
a10 <- Beta(paste(res_dir, "/lambda_1/beta.csv", sep = ""), "lambda 1.0")

expected <- Beta(paste(exp_dir, "beta_denovo.csv", sep = ""), "expected")

plot_list <- list(a00, a02, a04, a06, a08, a10)
x <- ggarrange(plotlist = plot_list)
ggarrange(expected, x, heights = c(1,2), ncol = 1)

#-------------------------------------------------------------------------------
# page 04-09 : alpha over iterations over lambdas
#-------------------------------------------------------------------------------
expected_alpha_path = paste(exp_dir, "expected_alpha.csv", sep = "")
a1 <- alpha_batch(paste(res_dir, "/lambda_0/alphas.csv", sep = ""), expected_alpha_path, 0)
a2 <- alpha_batch(paste(res_dir, "/lambda_0.2/alphas.csv", sep = ""), expected_alpha_path, 0.2)
a3 <- alpha_batch(paste(res_dir, "/lambda_0.4/alphas.csv", sep = ""), expected_alpha_path, 0.4)
a4 <- alpha_batch(paste(res_dir, "/lambda_0.6/alphas.csv", sep = ""), expected_alpha_path, 0.6)
a5 <- alpha_batch(paste(res_dir, "/lambda_0.8/alphas.csv", sep = ""), expected_alpha_path, 0.8)
a6 <- alpha_batch(paste(res_dir, "/lambda_1/alphas.csv", sep = ""), expected_alpha_path, 1)
a1
a2
a3
a4
a5
a6

#-------------------------------------------------------------------------------
# page 10 : likelihood over iterations over lambdas
#-------------------------------------------------------------------------------
b1 <- likelihood_iters(paste(res_dir, "/lambda_0/likelihoods.csv", sep = ""))
b2 <- likelihood_iters(paste(res_dir, "/lambda_0.2/likelihoods.csv", sep = ""))
b3 <- likelihood_iters(paste(res_dir, "/lambda_0.4/likelihoods.csv", sep = ""))
b4 <- likelihood_iters(paste(res_dir, "/lambda_0.6/likelihoods.csv", sep = ""))
b5 <- likelihood_iters(paste(res_dir, "/lambda_0.8/likelihoods.csv", sep = ""))
b6 <- likelihood_iters(paste(res_dir, "/lambda_1/likelihoods.csv", sep = ""))
lambda_labels <- c("Lambda 0.0", "Lambda 0.2", "Lambda 0.4", "Lambda 0.6", "Lambda 0.8", "Lambda 1")
lambda_plots <- ggarrange(b1, b2, b3, b4, b5, b6, labels = lambda_labels, vjust=12, hjust = -0.8)
lambda_plots

#-------------------------------------------------------------------------------
# page 11 : likelihood over lambdas
#-------------------------------------------------------------------------------
likelihood_lambdas(paste(res_dir, "/likelihoods.csv", sep = ""))

#-------------------------------------------------------------------------------
# THE END
#-------------------------------------------------------------------------------
dev.off()


