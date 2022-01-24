
#-------------------------------------------------------------------------------
# export to pdf
#-------------------------------------------------------------------------------

pdf(file="/home/azad/Documents/thesis/SigPhylo/data/results/myPlot.pdf")

M <- "/home/azad/Documents/thesis/SigPhylo/data/simulated/data_sigphylo.csv"
Phylogeny(M)

b_fixed <- "/home/azad/Documents/thesis/SigPhylo/data/simulated/beta_fixed.csv"
Beta(b_fixed)

b_denovo <- "/home/azad/Documents/thesis/SigPhylo/data/simulated/beta_denovo.csv"
Beta(b_denovo)

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

b1 <- likelihood_iters("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0/likelihoods.csv")
b2 <- likelihood_iters("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.2/likelihoods.csv")
b3 <- likelihood_iters("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.4/likelihoods.csv")
b4 <- likelihood_iters("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.6/likelihoods.csv")
b5 <- likelihood_iters("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.8/likelihoods.csv")
b6 <- likelihood_iters("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_1/likelihoods.csv")
lambda_labels <- c("Lambda 0.0", "Lambda 0.2", "Lambda 0.4", "Lambda 0.6", "Lambda 0.8", "Lambda 1")
lambda_plots <- ggarrange(b1, b2, b3, b4, b5, b6, labels = lambda_labels, vjust=12, hjust = -0.8)
lambda_plots

likelihood_lambdas("/home/azad/Documents/thesis/SigPhylo/data/results/likelihoods.csv")

dev.off()

joy <- "/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0/alphas.csv"
joyplot(joy)


#-------------------------------------------------------------------------------
# STORAGE
#-------------------------------------------------------------------------------





