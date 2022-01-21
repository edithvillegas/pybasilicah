
M <- "/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv"
M <- "/home/azad/Documents/thesis/SigPhylo/data/simulated/data_sigphylo.csv"
Phylogeny(M)

b <- "/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv"
Beta(b)

alpha_batch_path = "/home/azad/Documents/thesis/SigPhylo/data/results/lambda_1/alphas2.csv"
expected_alpha_path = "/home/azad/Documents/thesis/SigPhylo/data/simulated/expected_alpha.csv"
alpha_batch(alpha_batch_path, expected_alpha_path)
