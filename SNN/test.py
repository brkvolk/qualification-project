#############################################
#
# testing
#
#####################################################

import numpy as np
from SNN.network_parameters import T, I, H, delta_t, T, P_threshold, N
from SNN.spike_gen import gen_spike
from SNN.SRM_neuoron import neuron
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 8.0



spike1 = gen_spike()
spike2 = gen_spike()

input_spikes =[]
for i in range(H):
   input_spikes.append(gen_spike())

target = gen_spike()
weights = np.random.uniform(0, 0.2, H)

neuron2 = neuron()
neuron2.potential(input_spikes, weights)

print("output:", neuron2.output)
print("potential:", neuron2.data)



if neuron2.output: #если есть выход

    x1 = np.linspace(0, T, 1000)
    y1 = neuron2.data

    plt.plot(x1, y1, c="black")
    x2 = neuron2.output
    y2 = [P_threshold for i in x2]
    y3 = [P_threshold for i in x1]

    plt.scatter(x2, y2, c='orange')
    plt.plot(x1, y3, c="blue")
    plt.show()