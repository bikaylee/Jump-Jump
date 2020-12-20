from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

step_data = np.genfromtxt('steps.txt', delimiter="\t")
step_data.transpose()
steps = step_data[:,0]
sp_scores = step_data[:,1]
sp_distance = step_data[:,2]

episode_data = np.genfromtxt('episodes.txt', delimiter="\t")
episode_data.transpose()
episodes = episode_data[:,0]
ep_steps = episode_data[:,1]
ep_scores = episode_data[:,2]

def output(dataX, dataY, xLabel, yLabel):
    # 1. Rate of Change
    # 2. Max
    # 3. Median
    # 4. Mean
    # 5. Minimum
    
    print("=======================")
    print(xLabel + " vs " + yLabel + '\n')
    slope, intercept = np.polyfit(dataX, dataY, 1)
    print(round(slope,6))
    print(round(np.max(dataY,axis=0),4))
    print(round(np.median(dataY,axis=0),4))
    print(round(np.mean(dataY,axis=0),4))
    print(round(np.min(dataY,axis=0),4))
    print()
    
    # log_frequency = 100
    # box = np.ones(log_frequency) /log_frequency
    # Y_smooth = np.convolve(dataY, box, mode='same')
    # plt.clf()
    # plt.plot(dataX, dataY_smooth)
    # plt.plot(dataX, slope*dataX+intercept)

    
print("\nEnvironment 4")
output(steps, sp_scores, "Steps", "Scores")
output(steps, sp_distance, "Steps", "Relative Distance")
output(episodes, ep_steps, "Episodes", "Steps")
output(episodes, ep_scores, "Episodes", "Scores")
print("=======================")