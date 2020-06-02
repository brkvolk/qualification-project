#################################################################
#
#   SNN learning on randomly generated spike trains
#
#################################################################

import numpy as np
from SNN.spike_gen import gen_spike
from SNN.network import NeuralNetwork
from SNN.weight_initialization import weight_init
from SNN.network_parameters import I, H, ny, la, T
import matplotlib.pyplot as plt
target = gen_spike()
Wih, Who = weight_init()

input = []
i = 0
for i in range(I):
    input.append(gen_spike())

accurancy = 0.00001
max_epoch = 50
epoch = 0
NeuralNetwork = NeuralNetwork(input, target, Wih, Who)
#NeuralNetwork.ERROR()
E = NeuralNetwork.error
print (E)
while (epoch < max_epoch):
    # NeuralNetwork.feedforward()
    # NeuralNetwork.ERROR()
    while (E <= NeuralNetwork.error):
        NeuralNetwork.feedforward()

        NeuralNetwork.ERROR()
        Error = NeuralNetwork.error

        if (E > Error) :
            E = Error
            break
        else:
            NeuralNetwork.change_ny()

    if NeuralNetwork.error <= accurancy:
        NeuralNetwork.backprop()
        print("learning complited")
        break

    NeuralNetwork.backprop()
    epoch += 1

    print("Error = ", NeuralNetwork.error)
    print("output = ", NeuralNetwork.output)
    print("len: ", len(NeuralNetwork.output))
    print("target =", NeuralNetwork.target)
    print("len: ", len(NeuralNetwork.target))
   # print("Pot = ", NeuralNetwork.hiden_layer[2].data)
    print("epoch:", epoch)
    print("=================================================================================================================================================================")

y1 = [0 for i in NeuralNetwork.output]
y2 = [1 for i in target]
plt.scatter(NeuralNetwork.output, y1, c="black")
plt.scatter(target, y2, c="red")
plt.show()