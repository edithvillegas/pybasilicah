import utilities


seed = 1104
Tprofile = "A"
Iprofile = "X"
cos_path_org = "/home/azad/Documents/thesis/SigPhylo/data/cosmic/cosmic_catalogue.csv"
data = utilities.run_simulated(Tprofile, Iprofile, cos_path_org, seed)

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
'''

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




