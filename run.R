
#-------------------------------------------------------------------------------
# export to pdf
#-------------------------------------------------------------------------------

pdf(file="/home/azad/Documents/thesis/SigPhylo/data/results/myPlot.pdf")

#-------------------------------------------------------------------------------
# page 01 : phylogeny
#-------------------------------------------------------------------------------
M <- "/home/azad/Documents/thesis/SigPhylo/data/simulated/data_sigphylo.csv"
Phylogeny(M)

#-------------------------------------------------------------------------------
# page 02 : beta - fixed
#-------------------------------------------------------------------------------
b_fixed <- "/home/azad/Documents/thesis/SigPhylo/data/simulated/beta_fixed.csv"
Beta(b_fixed)

#-------------------------------------------------------------------------------
# page 03 : beta - denovo vs inferred
#-------------------------------------------------------------------------------
a00 <- ggarrange(Beta("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0/beta.csv"), 
                 Beta("/home/azad/Documents/thesis/SigPhylo/data/simulated/beta_denovo.csv"))
a02 <- ggarrange(Beta("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.2/beta.csv"), 
                 Beta("/home/azad/Documents/thesis/SigPhylo/data/simulated/beta_denovo.csv"))
a04 <- ggarrange(Beta("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.4/beta.csv"), 
                 Beta("/home/azad/Documents/thesis/SigPhylo/data/simulated/beta_denovo.csv"))
a06 <- ggarrange(Beta("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.6/beta.csv"), 
                 Beta("/home/azad/Documents/thesis/SigPhylo/data/simulated/beta_denovo.csv"))
a08 <- ggarrange(Beta("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0.8/beta.csv"), 
                 Beta("/home/azad/Documents/thesis/SigPhylo/data/simulated/beta_denovo.csv"))
a10 <- ggarrange(Beta("/home/azad/Documents/thesis/SigPhylo/data/results/lambda_1/beta.csv"), 
                 Beta("/home/azad/Documents/thesis/SigPhylo/data/simulated/beta_denovo.csv"))
a00
a02
a04
a06
a08
a10

#-------------------------------------------------------------------------------
# page 04 : alpha over iterations over lambdas
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
# page 05 : likelihood over iterations over lambdas
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
# page 06 : likelihood over lambdas
#-------------------------------------------------------------------------------
likelihood_lambdas("/home/azad/Documents/thesis/SigPhylo/data/results/likelihoods.csv")


#-------------------------------------------------------------------------------
# THE END
#-------------------------------------------------------------------------------
dev.off()






e <- "/home/azad/Documents/thesis/SigPhylo/data/results/lambda_0/alpha.csv"
df <- as.data.frame(read.table(e, sep = ",", header = FALSE))
typeof(df)
df <- t(df)
typeof(df)
view(df)

plot <- ggplot(data=df, aes(x=V1)) + 
  geom_bar() +
  ggtitle("Likelihood changes Over Lambdas") + 
  xlab("Lambdas") + 
  ylab("Likelihood")
  






