import pandas as pd
import basilica
import simulation

M_path = "/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv"
B_input_path = "/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv"
cosmic_path = "/home/azad/Documents/thesis/SigPhylo/data/cosmic/cosmic_catalogue.csv"

M = pd.read_csv(M_path)
B_input = pd.read_csv(B_input_path, index_col=0)
cosmic_df = pd.read_csv(cosmic_path, index_col=0)
k_list = [0, 1, 2, 3, 4, 5]
fixedLimit = 0.05
denovoLimit = 0.9

#A_inf_df, B_inf_fixed_df, B_inf_denovo_df = BaSiLiCa(M, B_input, k_list, cosmic_df, fixedLimit, denovoLimit)

#print("Alpha:\n",A_inf_df)
#print("Beta Fixed:\n", B_inf_fixed_df)
#print("Beta Denovo", B_inf_denovo_df)

seed = 23
output = simulation.run_simulated("A", "X", cosmic_path, fixedLimit, denovoLimit, seed)
print(output["Tprofile"])
print(type(output["Tprofile"]))
print(output["Iprofile"])
print(type(output["Iprofile"]))