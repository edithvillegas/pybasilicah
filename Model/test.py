
import dependences
import model
import svi

my_path = "/Users/riccardobergamin/Desktop/PycharmProjects/SigPhylo/data/"
data_file = "data_sigphylo.csv"
aging_file = "beta_aging.csv"

# load data
M = pd.read_csv(my_path + data_file)
beta_aging = pd.read_csv(my_path + aging_file)

# get counts and contexts
M_counts = get_phylogeny_counts(M)
beta_counts,signature_names,contexts = get_signature_profile(beta_aging)

# define adjacency matrix
A = torch.tensor([[1,1,0,0,0],[1,1,1,1,0],[0,1,1,1,0],[0,1,1,1,1],[0,0,0,1,1]])

params = {"k_denovo" : 1, "beta_fixed" : beta_counts, "A" : A, "lambda": 0.2}

params = full_inference(M_counts,params,lr = 0.05,steps_per_iteration = 500,num_iterations = 10)





