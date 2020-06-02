########################################################################################################################
#
#   functions for encoding figits into spikes
#
########################################################################################################################
import numpy as np
import random
from SNN.network_parameters import T
import cv2

#encode digits with frequanses between 30 & 50 Hz by Poisson process

def digits_to_spikes():
   D = 10 # number of digits
   digits = []
   for i in range(D):
       digits.append([])

   print (digits)

   la = 30

   for i in range(10):
        t = 0
        while t < T:
            E = random.expovariate(la)
            t = t + E / la
            digits[i].append(t)
        la += 2


   return digits




digits = digits_to_spikes()
print (digits)
print (digits[2])

