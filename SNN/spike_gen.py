##################################################################################
#
# Functions for spikes generating with frequency la
#
#################################################################################

import numpy as np
import random
from SNN.network_parameters import T

def gen_spike_with_length(la): #генерируем спайки пуассоновским процессом
    spike = []
    t = 0
    k = 0
    while t < T:
        E = random.expovariate(la)
        k = k + 1
        t = t + E / la
        spike.append(t)

    return spike, k

def gen_spike(la): #генерируем спайки пуассоновским процессом
    spike = []
    t = 0
    k = 0
    m = T / (la + 1)
    while t < T:

        E = random.expovariate(la)
        if m < E/la:
            t = t + E / la
            k = k + 1
        else: continue
        if t >= T or k == int_r(la):
            return spike
        spike.append(t)
        t += (m - E/la)
    return spike

def int_r(num):
    num = int(num + (0.5 if num > 0 else -0.5))
    return num

# def gen_spike(la):
#     spike = []
#
#     num_spikes_per_cell = int_r(la)
#     frequency = int_r(la)
#
#     isi = np.random.poisson(frequency, num_spikes_per_cell)
#
#     spike =np.interp(np.sort(isi), [min(isi)-1,max(isi)+1],[0, T] ).tolist()
#
#     return spike