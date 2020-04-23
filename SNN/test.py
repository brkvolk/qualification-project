#############################################
#
# testing
#
#####################################################from
import numpy as np
from SNN.spike_gen import gen_spike
from SNN.SRM_neuoron import neuron
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 8.0
T = 1
dt = 0.001

spike1 = gen_spike()
spike2 = gen_spike()

input_spikes =[]
for i in range(5):
 input_spikes.append(gen_spike())

target = gen_spike()
#weights = np.random.uniform(0, 0.2,  2)
weights = np.random.uniform(0, 0.2,  5)
#initial_weights1 = np.random.uniform(0, 0.2, (I, H))
#initial_weights2 = np.random.uniform(0, 0.2, H)
#neuron1 = neuron()
neuron2 = neuron()

#neuron1.potential([spike1, spike2], weights)
neuron2.potential(input_spikes, weights)

#print(neuron1.output)
print(neuron2.output)

#print(neuron1.data)
print(neuron2.data)



if not not neuron2.output: #если есть выход
   # fig, (ax1, ax2 ) = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
#ax1.set_title('potential')
#ax2.set_xlabel('output')
    x1 = np.linspace(0, 1, 1000)
    y1 = neuron2.data
    #ax1.stem(x1, y1, linefmt='black', markerfmt = 'None', basefmt='None')
    plt.plot(x1, y1, c="black")
    x2 = neuron2.output
    y2 = [1 for i in x2]
   #ax2.stem(x2, y2, linefmt='black', markerfmt = 'None', basefmt='None')
    plt.scatter(x2, y2, c='orange')
    plt.show()