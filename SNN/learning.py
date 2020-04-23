#################################################################
#
#   SNN learning on randomly generated spike trains
#
#################################################################

import numpy as np
from SNN.spike_gen import gen_spike
from SNN.network import NeuralNetwork
from SNN.weight_initialization import wehgt_init
from SNN.network_parameters import I, H, ny, la, T

target = gen_spike()
Wih, Who = wehgt_init()

input = []
i = 0
for i in range(I):
    input.append(gen_spike())

max_epoch = 100
epoch = 0
NeuralNetwork = NeuralNetwork(input, target, Wih, Who)
while (epoch < max_epoch):
    NeuralNetwork.feedforward()
    NeuralNetwork.ERROR()
    if NeuralNetwork.error <= 0.1:
        break
    NeuralNetwork.backprop()
    epoch += 1
    print("Error = ", NeuralNetwork.error)
    print ("output = ", NeuralNetwork.output)
    print("target =", NeuralNetwork.target)
    print("=================================================================================================================================================================")