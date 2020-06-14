import numpy as np
import cv2
from numpy import interp
from SNN.network_parameters import T, la
from SNN.spike_gen import gen_spike
import matplotlib.pyplot as plt
from SNN.weight_initialization import weight_init
import pandas as pd

path = "C:\\Users\\ALEX\\PycharmProjects\\3-neuron_network\\SNN\\iris\\"
max_epoch = 100
setosa_error, versicolor_error, virginica_error = np.load(path+'setosa_error.npy'), np.load(path+'versicolor_error.npy'), np.load(path+'virginica_error.npy')
np.array(setosa_error).tolist()
np.array(versicolor_error).tolist()
np.array(virginica_error).tolist()

x = [i for i in range(max_epoch)]

y1 = [setosa_error[i] for i in x]
y2 = [versicolor_error[i] for i in x]
y3 = [virginica_error[i] for i in x]

plt.plot(x, y1, color='blue')
plt.plot(x, y2, color='red')
plt.plot(x, y3, color='green')

plt.show()