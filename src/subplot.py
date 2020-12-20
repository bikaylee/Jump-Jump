import numpy as np 
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(11, 8))

def plotGraph(fileName, xLabel, yLabel, category, i, j):
    data = np.genfromtxt(fileName+'.txt', delimiter="\t")
    log_frequency = 100
    box = np.ones(log_frequency) /log_frequency

    X = []
    Y = []

    for item in data:
        X.append(item[0])
        Y.append(item[category])

    Y_smooth = np.convolve(Y, box, mode='same')
    ax[i][j].plot(X, Y_smooth)
    ax[i][j].set_ylabel(yLabel)
    ax[i][j].set_xlabel(xLabel)

plotGraph("steps", "Steps", "Scores", 1, 0, 0)
plotGraph("steps","Steps", "Relative Distances", 2, 0, 1)
plotGraph("episodes", "Episodes", "Steps", 1, 1, 0)
plotGraph("episodes", "Episodes", "Scores", 2, 1, 1)
    
fig.suptitle("Environment 4", fontsize=16)
plt.subplots_adjust(hspace=0.4, wspace=0.2)
plt.show()    
fig.savefig("Environment 4.png")