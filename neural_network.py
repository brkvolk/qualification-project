import numpy as np
import random
import math
import functions
"""""""""""
струтрура для нейронной сети
"""""""""""
from functions import gen_spike, d_spike, Who, dwoh, Wih, dwhi, Error, spikeIP, timeIP, neuron_state, spike1, m, k,  multipl, nu, sigmoid, I, H

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x# список входных спайков
        self.weights1   = np.random.uniform(0, 0.2, (I, H))
        self.weights2   = np.random.uniform(0, 0.2, H)
        self.y          = y#желаемый спайк
        self.output     = np.zeros(y.shape)#куальный выход - должен быть размера у?

    def feedforward(self):
       self.layer_hiden = multipl(self.input, self.weights1)#взвешиваем кортежи спайков -  их надо сначала прогнать ч\з потенциал? - получаем вектор из списоков списков(АААААААААААААААА)
       #вектор списков, выходящий со скрытого слоя(чз потенциал):
       self.layer_hiden = neuron_state(t, self.layer_hiden,  self.weights2)#что-то не так
       #self.output = sigmoid(multipl(self.layer_h, self.weights2))  #
       self.output = multipl(self.layer_hiden, self.weights2)#взвешиваем и суммируем

       #Error

    def backprop(self):
         #d_weights2 = np.dot(self.layer_h.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
         d_weights2 =-nu*(spikeIP(self.output,self.layer_hiden)-spikeIP(self.y-self.layer_hiden))
         #d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
          #                                         self.weights2.T) * sigmoid_derivative(self.layer1)))
         d_weights1 = -nu(spikeIP(self.output,self.input)- spikeIP(self.y,self.input)) * self.weights2

          # update the weights with the derivative (slope) of the loss function
         self.weights1 += d_weights1
         self.weights2 += d_weights2