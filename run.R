#library(reticulate)

#-------------------------------------------------------------------------------
# export to pdf
#-------------------------------------------------------------------------------

pdf(file="/home/azad/Documents/thesis/SigPhylo/data/results/myPlot.pdf")

#-------------------------------------------------------------------------------
# page 01 : phylogeny & beta fixed
#-------------------------------------------------------------------------------
M <- "/home/azad/Documents/thesis/SigPhylo/data/simulated/data_sigphylo.csv"
m_plot <- Phylogeny(M)

b_fixed <- "/home/azad/Documents/thesis/SigPhylo/data/simulated/beta_fixed.csv"
fixed_plot <- Beta(b_fixed, "fixed beta")

plot_list <- list(m_plot, fixed_plot)
ggarrange(plotlist = plot_list, ncol = 1)

#-------------------------------------------------------------------------------
# page 02 : beta (denovo vs inferred)
#-------------------------------------------------------------------------------
a00 <- Beta("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0/beta.csv", "lambda 0")
a02 <- Beta("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.2/beta.csv", "lambda 0.2")
a04 <- Beta("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.4/beta.csv", "lambda 0.4")
a06 <- Beta("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.6/beta.csv", "lambda 0.6")
a08 <- Beta("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.8/beta.csv", "lambda 0.8")
a10 <- Beta("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_1/beta.csv", "lambda 1.0")

expected <- Beta("/home/azad/Documents/thesis/SigPhylo/data/simulated/beta_denovo.csv", "expected")

plot_list <- list(a00, a02, a04, a06, a08, a10)
x <- ggarrange(plotlist = plot_list)
ggarrange(expected, x, heights = c(1,2), ncol = 1)

#-------------------------------------------------------------------------------
# page 03 : alpha over iterations over lambdas
#-------------------------------------------------------------------------------
expected_alpha_path = "/home/azad/Documents/thesis/SigPhylo/data/simulated/expected_alpha.csv"
a1 <- alpha_batch("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0/alphas.csv", expected_alpha_path, 0)
a2 <- alpha_batch("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.2/alphas.csv", expected_alpha_path, 0.2)
a3 <- alpha_batch("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.4/alphas.csv", expected_alpha_path, 0.4)
a4 <- alpha_batch("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.6/alphas.csv", expected_alpha_path, 0.6)
a5 <- alpha_batch("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.8/alphas.csv", expected_alpha_path, 0.8)
a6 <- alpha_batch("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_1/alphas.csv", expected_alpha_path, 1)
a1
a2
a3
a4
a5
a6

#-------------------------------------------------------------------------------
# page 04 : likelihood over iterations over lambdas
#-------------------------------------------------------------------------------
b1 <- likelihood_iters("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0/likelihoods.csv")
b2 <- likelihood_iters("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.2/likelihoods.csv")
b3 <- likelihood_iters("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.4/likelihoods.csv")
b4 <- likelihood_iters("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.6/likelihoods.csv")
b5 <- likelihood_iters("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.8/likelihoods.csv")
b6 <- likelihood_iters("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_1/likelihoods.csv")
lambda_labels <- c("Lambda 0.0", "Lambda 0.2", "Lambda 0.4", "Lambda 0.6", "Lambda 0.8", "Lambda 1")
lambda_plots <- ggarrange(b1, b2, b3, b4, b5, b6, labels = lambda_labels, vjust=12, hjust = -0.8)
lambda_plots

#-------------------------------------------------------------------------------
# page 05 : likelihood over lambdas
#-------------------------------------------------------------------------------
likelihood_lambdas("/home/azad/Documents/thesis/SigPhylo/data/results/likelihoods.csv")


#-------------------------------------------------------------------------------
# THE END
#-------------------------------------------------------------------------------
dev.off()


