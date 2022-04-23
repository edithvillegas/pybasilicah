from PyBaSiLiCa.PyBaSiLiCa import utilities
from PyBaSiLiCa.test import simulation


def func():
    cosmic_path = "/home/azad/Documents/thesis/SigPhylo/cosmic/cosmic_catalogue.csv"
    cosmic_df, denovo_df = utilities.cosmic_denovo(cosmic_path)
    for Tprofile in ["A", "B", "C"]:
        for Iprofile in ["X", "Y", "Z"]:
            for iter in range(4):
                simulation.run_simulated(Tprofile, Iprofile, cos_path_org, fixedLimit, denovoLimit, seed)

                #print("num samples:", len(list(alpha_df.index)))
                #print("num Total signatures:", len(list(alpha_df.columns)))
                #print("num fixed signatures:", len(list(beta_fixed_df.index)))
                #print("num denovo signatures:", len(list(beta_denovo_df.index)))
                #print(beta_input)
                #print("==========================================================")







cos_path_org = "/home/azad/Documents/thesis/SigPhylo/data/cosmic/cosmic_catalogue.csv"
fixedLimit = 0.05
denovoLimit = 0.9

Tprofile = "A"
Iprofile = "X"
seed = 1104



def simRun(Tprofile, Iprofile, cos_path_org, seed):
    data = utilities.run_simulated(Tprofile, Iprofile, cos_path_org, seed)
    return data

    '''
    output = {
        "M"                 : M,                    # dataframe
        "A_target"          : A,                    # dataframe
        "B_fixed_target"    : B_fixed,              # dataframe
        "B_denovo_target"   : B_denovo,             # dataframe
        "B_input"           : B_input,              # dataframe
        "A_inf"             : A_inf,                # dataframe
        "B_fixed_inf"       : B_fixed_inf,          # dataframe
        "B_denovo_inf"      : B_denovo_inf_labeled, # dataframe

        "GoodnessofFit"     : gof,                      # float
        "Accuracy"          : B_fixed_accuracy,         # float
        "Quantity"          : B_denovo_quantity,        # bool
        "Quality"           : B_denovo_quality,         # float
        }
    

    print("========================================================")
    #print("Alpha Target\n",         data["A_target"])
    print("Beta Fixed Target",      list(data["B_fixed_target"].index))
    print("Beta Denovo Target",     list(data["B_denovo_target"].index))
    print("Beta Input",             list(data["B_input"].index))
    print("Alpha Inferred",         list(data["A_inf"].columns))
    print("Beta Fixed Inferred",    list(data["B_fixed_inf"].index))
    print("Beta Denovo Inferred",   list(data["B_denovo_inf"].index))

    print("Goodness of Fit:",       data["GoodnessofFit"])
    print("Beta Fixed Accuarcy:",   data["Accuracy"])
    print("Beta Denovo Quantity:",  data["Quantity"])
    print("Beta Denovo Quality:",   data["Quality"])
    '''




