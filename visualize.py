import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import aux

def catalogue(m):
    fig = plt.figure(figsize=(8,8)) #  figure size (optional)
    for i in range(3):
        xpoints = range(0, 96)
        ypoints = np.array(m.iloc[i])
        plt.subplot(3,1,i+1)  #  subplot 1
        plt.bar(xpoints, ypoints)

        #plt.bar(xpoints, ypoints)
        #plt.title("genome catalogue")
        #plt.xlabel("mutation features")
        #plt.ylabel("number of mutations")
        #plt.grid()
        #plt.legend()
    plt.show()
                  #  draw subplot group

def alpha_convergence(infered_alpha, expected_alpha, branch):

    n = expected_alpha.shape[0]             # no. of branches
    k = expected_alpha.shape[1]             # no. of signatures
    itr = int(infered_alpha.shape[0] / n)   # no. of iterations

    xpoints = np.array(range(itr))

    legends = []
    for l in range(k):
        legends.append("alpha" + str(l+1))

    # iterate over signature profile
    for i in range(k):
        r = expected_alpha[branch][i]
        values = []

        for j in range(branch, infered_alpha.shape[0], n):
            value = float("{:.3f}".format(infered_alpha.iloc[j][i] / r))
            values.append(value)
        
        ypoints = np.array(values)
        plt.plot(xpoints, ypoints, label=legends[i])

    plt.title("alpha convergence")
    plt.xlabel("iterations")
    plt.ylabel("alpha ratio")
    plt.grid()
    plt.legend()
    plt.show()


# not completed
def beta(b):

    k = b.shape[0]             # no. of signatures

    print(k)

    xpoints = np.array(range(96))

    for i in range(k):
        ypoints = np.array(b.iloc[i])
        plt.bar(xpoints, ypoints)
    
    plt.show()

'''
#for i in range(n):
#plt.subplot(1, n, i+1)
#plt.plot(xpoints, ypoints, label=legends[j]+" : real = "+str(format(target[branch][j].item(), '.2f'))+")")
print(
    "real :", r, 
    "infered :", infered_alpha.iloc[t][signature], 
    "ratio :", float("{:.3f}".format(infered_alpha.iloc[t][signature] / r))
    )   
'''