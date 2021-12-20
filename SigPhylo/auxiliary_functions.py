

# load data

def get_phylogeny_counts(M):

    M = M.values
    M = torch.tensor(np.array(M,dtype=float))
    M = M.float()

    return M

def get_signature_profile(beta):

    contexts = list(beta.columns[1:])
    signature_names = list(beta.values[:, 0])
    counts = beta.values[:,1:]
    counts = torch.tensor(np.array(counts, dtype=float))
    counts = counts.float()

    return counts,signature_names,contexts


