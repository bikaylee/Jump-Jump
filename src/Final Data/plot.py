import numpy as np 
import matplotlib.pyplot as plt


def plotGraph(fileName, xLabel, yLabel):

	log_frequency = 50
	box = np.ones(log_frequency) /log_frequency
	data = np.genfromtxt(fileName+'.txt', delimiter="	")

	X = []
	Y = []

	for item in data:
		X.append(item[0])
		Y.append(item[1])
		if len(Y) > log_frequency and len(Y) % log_frequency == 0:
			Y_smooth = np.convolve(Y, box, mode='same')
			plt.clf()
			plt.plot(X, Y_smooth)
			plt.title('Pixel Jump')
			plt.ylabel(yLabel)
			plt.xlabel(xLabel)
			plt.savefig(fileName+'.png')


plotGraph("returns", "Steps", "Return")
plotGraph("differences_returns","Steps", "Differences")
plotGraph("distance_returns", "Episodes", "Distances")


