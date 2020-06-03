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


input = []
i = 0
for i in range(I):
    input.append(gen_spike())

Wih, Who = weight_init()
print("I want", Wih)
print("\n", Who)

NeuralNetwork1 = NeuralNetwork(input, [], Wih, Who)
NeuralNetwork1.feedforward()
target = NeuralNetwork1.output


Wih, Who = weight_init()
accurancy = 0.000001
max_epoch = 500
epoch = 0
NeuralNetwork = NeuralNetwork(input, target, Wih, Who)
# NeuralNetwork.ERROR()
# E = NeuralNetwork.error
# print (E)
Error=[]

while (epoch < max_epoch):
    NeuralNetwork.feedforward()
    NeuralNetwork.ERROR()
    Error.append(NeuralNetwork.error)
    # while (E <= NeuralNetwork.error):
    #     NeuralNetwork.feedforward()
    #
    #     NeuralNetwork.ERROR()
    #     Error = NeuralNetwork.error
    #
    #     if (E > Error) :
    #         E = Error
    #         break
    #     else:
    #         NeuralNetwork.change_ny()

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
    print("epoch:", epoch)
    print("=================================================================================================================================================================")
print("learning complited")

print("Ive got", NeuralNetwork.weights1)
print("\n", NeuralNetwork.weights2)
y1 = [0 for i in NeuralNetwork.output]
y2 = [1 for i in target]
y3 = Error
x3 = [i for i in range(50)]
plt.scatter(NeuralNetwork.output, y1, c="black")
plt.scatter(target, y2, c="red")
# plt.plot(x3, y3, c="blue")
plt.show()