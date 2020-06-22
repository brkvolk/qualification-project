####################################################################################
#
# Structure of neural network + backprop alg methods
#
##################################################################################

import numpy as np
from SNN.inner_products import spikeIP, timeIP
from SNN.network_parameters import I, H, ny, L
from SNN.SRM_neuoron import neuron

class NeuralNetwork:
    def __init__(self, input, target, Wih, Who):
        self.input              = input         # список входных спайков
        self.weights1           = Wih           #matrix
        self.weights2           = Who           #vector
        self.hiden_layer = [neuron() for i in range(H)]#скрытый слой из H нейронов
        self.output_layer       = neuron()#выходной слой их одного нейрона
        self.target             = target         #желаемый выход - а если мы классифицируем, а не обучаем?
        self.output             = []             #реальный выход
        self.error: float       = 100.
        self.post_hiden_spikes  = []
        self.ny                 = ny#скорость обучения

    def feedforward(self):
        self.post_hiden_spikes = []
        for h in range(H):
            self.hiden_layer[h].potential(self.input, self.weights1[:, h])      #список списков  - выходов со скрытых нейронов
            self.post_hiden_spikes.append(self.hiden_layer[h].output)

        self.output_layer.potential(self.post_hiden_spikes, self.weights2 )  #т.к. выходной нейрон один, то цикла нет
        self.output = self.output_layer.output

    def ERROR(self):
        self.error = 0.5 * (spikeIP(self.output, self.output) - 2 * (spikeIP(self.output, self.target)) + spikeIP(self.target, self.target))

    def backprop(self):
        d_weights1 = np.zeros((I, H), dtype=float)
        d_weights2 = np.zeros(H, dtype=float)

        h = 0
        while h < H:
            # print(self.output)

            d_weights2[h] = -self.ny * (spikeIP(self.output, self.post_hiden_spikes[h]) - spikeIP(self.target, self.post_hiden_spikes[h]))  #вектор

            i = 0
            while i < I:
                d_weights1[i][h] = -self.ny * (spikeIP(self.output, self.input[i]) - spikeIP(self.target, self.input[i])) * self.weights2[h] #матрица

                i += 1
            h += 1

        print(self.weights1,"\n", d_weights1, "\n" )
        print(self.weights2, "\n", d_weights2, "\n")

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def quantization(self):
        w1max = np.amax(abs(self.weights1), axis=1)
        w2max = np.amax(abs(self.weights2))
        if L >= 2:
            d1 = w1max/(L-1)
            d2 = w2max/(L-1)
        else:
            print("wrong L")
            return
        for i in range(I):
            for j in range(H):
                for k in range(1, L):
                    if ((k - 1) * d1[i] < self.weights1[i][j]) & (self.weights1[i][j] < k * d1[i]):
                        self.weights1 = d1[i] * (k - 1) * np.sign(self.weights1)
                    if ((k - 1) * d2 < self.weights2[j]) & (self.weights2[j] < k * d2):
                        self.weights2 = d2 * (k - 1) * np.sign(self.weights2)
