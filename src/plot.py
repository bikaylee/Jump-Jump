import numpy as np 
import matplotlib.pyplot as plt

def plotGraph(fileName, xLabel, yLabel, category):
	log_frequency = 50
	box = np.ones(log_frequency) /log_frequency
	data = np.genfromtxt(fileName+'.txt', delimiter="\t")

	X = []
	Y = []

	for item in data:
		X.append(item[0])
		Y.append(item[category])
		if len(Y) > log_frequency and len(Y) % log_frequency == 0:
			Y_smooth = np.convolve(Y, box, mode='same')
			plt.clf()
			plt.plot(X, Y_smooth)
			plt.title('Pixel Jump')
			plt.ylabel(yLabel)
			plt.xlabel(xLabel)
			plt.savefig(xLabel + ' vs ' + yLabel +'.png')

plotGraph("steps", "Steps", "Scores", 1)
plotGraph("steps","Steps", "Relative Differences", 2)
plotGraph("episodes", "Episodes", "Steps", 1)
plotGraph("episodes", "Episodes", "Scores", 2)

