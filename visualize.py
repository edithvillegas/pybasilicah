import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import aux

# visualize the alpha values among iterations
# 1st arg : list of alphas
# 2nd arg : branch number (starts from 1)

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


def catalogue(m):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    #ax.set_xlim(0, 96)
    x = range(0, 96)
    y = m[0]
    #ax.bar(x, y)
    plt.bar(x, y, width= 0.9, align='center',color='cyan')

    ax.set_title("genome catalogue")
    ax.set_xlabel('mutation features')
    ax.set_ylabel('number of mutations')
    ax.set_xticks([0, 15, 31, 47, 63, 79, 95])
    ax.set_xticklabels([1, 16, 32, 48, 64, 80, 96])
    #plt.legend(labels = ['Total mutations'])
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