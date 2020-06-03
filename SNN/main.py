import numpy as np
from SNN.spike_gen import gen_spike
from SNN.network import NeuralNetwork
from SNN.weight_initialization import weight_init
from SNN.network_parameters import I, H, ny, la, T
import matplotlib.pyplot as plt


def learning(input, target, max_epoch, Wih, Who, accurancy):
    NeuralNetwork = NeuralNetwork(input, target, Wih, Who)
    epoch=0
    while(epoch<=max_epoch):
        NeuralNetwork.feedforward()
        NeuralNetwork.ERROR()
        if NeuralNetwork.error <= accurancy:   #а это надо?
            NeuralNetwork.backprop()
            print("learning complited")
            break

        NeuralNetwork.backprop()
        epoch += 1


def recognizing(input, Wih, Who):
    NeuralNetwork = NeuralNetwork(input, [], Wih, Who)
    NeuralNetwork.feedforward()
    result = NeuralNetwork.output
    return(result)



#np.save("results/Wih",NeuralNetwork.weights1)
#np.save("results/Who",NeuralNetwork.weights2)

#np.savez("results/weights",NeuralNetwork.weights1=wih ,NeuralNetwork.weights2=who)
#data = np.load('results/weights.npz')
#data.files

#data.close()


#NeuralNetwork.weights1 = np.load('results\Wih.npy')
