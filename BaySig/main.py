from typing import Counter
import utilities


cos_path = "/home/azad/Documents/thesis/SigPhylo/cosmic/cosmic_catalogue.csv"

data = utilities.run_simulated(cos_path)

print("alpha_target\n", data["alpha_target"])
print("beta_target\n", data["beta_target"])
print("beta_fixed\n", data["beta_fixed"])
print("alpha_inferred\n", data["alpha_inferred"])
print("beta_inferred\n", data["beta_inferred"])
print("Accuracy", data["Accuracy"])
print("Precision", data["Precision"])
print("Recall", data["Recall"])
print("F1", data["F1"])
print("alpha_mse", data["alpha_mse"])
print("GoF", data["GoF"])

'''
#------------------------------------------------------------------------------------
# create new directory (overwrite if exist) and export data as JSON file
#------------------------------------------------------------------------------------
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
#print("New directory made!")

with open(save_path + "/output.json", 'w') as outfile:
    json.dump(output, outfile, cls=utilities.NumpyArrayEncoder)
    #print("Exported as JSON file!")
    
return output
'''


