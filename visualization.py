import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize(alphas, branch):

    print("visualizing alphas in branch", branch)

    Ralpha = torch.tensor([
        [0.35, 0.50, 0.15],
        [0.52, 0.43, 0.05],
        [0.51, 0.45, 0.04],
        [0.02, 0.03, 0.95],
        [0.23, 0.46, 0.31]
        ])

    n = Ralpha.size()[0]
    m = Ralpha.size()[1]

    xpoints = np.array(range(len(alphas)))
    legends = ["sig1", "sig2", "sig3"]

    #for i in range(n):
        #plt.subplot(1, n, i+1)

    i=0

    for j in range(m):

        r = Ralpha[i][j].item()

        vals = []
        for k in range(len(alphas)):
            c = alphas[k][i][j].item()
            vals.append(c)

        ypoints = np.array(vals)

        plt.plot(xpoints, ypoints, label=legends[j])


    plt.title("alpha change over iterations")
    plt.xlabel("iterations")
    plt.ylabel("alpha value")
    plt.grid()
    plt.legend()
    plt.show()


def test(alphas, j, branch):
    c = alphas[j][branch-1][1].item()
    print(c)