#library(reticulate)

res_dir <- "/home/azad/Documents/thesis/SigPhylo/data/results/output"
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

plot_list = list()
files <- list.files(res_dir, pattern = "K_*", full.names = TRUE)
for (currentFile in files) {
  lambdas <- list.files(currentFile, pattern = "lambda*", full.names = TRUE)
  lambda <- read.table(lambdas, header = FALSE)
  plot_list[[length(plot_list)+1]] <- signature_share(
    paste(currentFile, "/alpha.csv", sep = ""), 
    paste("lambda=", lambda[[1]])
    )
}
x <- ggarrange(plotlist = plot_list)
ggarrange(exp, x, ncol = 1)

#-------------------------------------------------------------------------------
# page 03 : beta (denovo vs inferred)
#-------------------------------------------------------------------------------
expected <- Beta(paste(exp_dir, "beta_denovo.csv", sep = ""), "expected")
x <- Beta("/home/azad/Documents/thesis/SigPhylo/data/results/output/K_3_L_0/beta.csv", "best")

plot_list = list()
files <- list.files(res_dir, pattern = "K_*", full.names = TRUE)
for (currentFile in files) {
  lambdas <- list.files(currentFile, pattern = "lambda*", full.names = TRUE)
  lambda <- read.table(lambdas, header = FALSE)
  plot_list[[length(plot_list)+1]] <- Beta(
    paste(currentFile, "/beta.csv", sep = ""), 
    paste("lambda=", lambda[[1]])
  )
}
x <- ggarrange(plotlist = plot_list, ncol = 1)
ggarrange(expected, x, heights = c(1,2), ncol = 1)

#-------------------------------------------------------------------------------
# page 04-09 : alpha over iterations over lambdas
#-------------------------------------------------------------------------------
expected_alpha_path = paste(exp_dir, "expected_alpha.csv", sep = "")

files <- list.files(res_dir, pattern = "K_*", full.names = TRUE)
for (currentFile in files) {
  lambdas <- list.files(currentFile, pattern = "lambda*", full.names = TRUE)
  lambda <- read.table(lambdas, header = FALSE)[[1]]
  alpha_batch(paste(currentFile, "/alphas.csv", sep = ""), expected_alpha_path, paste("lambda=", lambda))
  print(currentFile)
  print(lambdas)
  print(lambda)
  }

alpha_batch(paste(currentFile, "/alphas.csv", sep = ""), expected_alpha_path, paste("lambda =", lambda))


alpha_batch_path <- "/home/azad/Documents/thesis/SigPhylo/data/results/new/K_1_L_0/alphas.csv"
expected_alpha_path <- "/home/azad/Documents/thesis/SigPhylo/data/simulated/expected_alpha.csv"
lbl <- "hello"
alpha_batch <- read.table(alpha_batch_path, sep = ",", header = FALSE)
expected_alpha <- read.table(expected_alpha_path, sep = ",", header = FALSE)

a <- as.vector(t(data.matrix(expected_alpha)))
zero <- which(a==0)
b <- as.data.frame(matrix(a, ncol=length(a), byrow = T))
w <- do.call("rbind", replicate(nrow(alpha_batch), b, simplify = FALSE))
ratio <- alpha_batch / w

ratio[zero] <- (1 - alpha_batch[zero])
ratio$itr <- 0:(nrow(alpha_batch)-1)

df <- melt(ratio, id.vars=c("itr"), variable.name = "alpha", 
           value.name = "value")


#a1 <- alpha_batch(paste(res_dir, "/lambda_0/alphas.csv", sep = ""), expected_alpha_path, 0)
#a2 <- alpha_batch(paste(res_dir, "/lambda_0.2/alphas.csv", sep = ""), expected_alpha_path, 0.2)
#a3 <- alpha_batch(paste(res_dir, "/lambda_0.4/alphas.csv", sep = ""), expected_alpha_path, 0.4)
#a4 <- alpha_batch(paste(res_dir, "/lambda_0.6/alphas.csv", sep = ""), expected_alpha_path, 0.6)
#a5 <- alpha_batch(paste(res_dir, "/lambda_0.8/alphas.csv", sep = ""), expected_alpha_path, 0.8)
#a6 <- alpha_batch(paste(res_dir, "/lambda_1/alphas.csv", sep = ""), expected_alpha_path, 1)
#a1
#a2
#a3
#a4
#a5
#a6

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


