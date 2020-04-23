##################################################################################
#
# Functions for spikes generating
#
#################################################################################

import numpy as np
import random
from SNN.network_parameters import T, la

def gen_spike_with_length(): #генерируем спайки пуассоновским процессом
    spike = []
    t = 0
    k = 0
    while t < T:
        E = random.expovariate(la)
        k = k + 1
        t = t + E / la
        spike.append(t)

    #return np.array(spike), k
    return spike, k
def gen_spike(): #генерируем спайки пуассоновским процессом
    spike = []
    t = 0
    k = 0
    while t < T:
        E = random.expovariate(la)
        k = k + 1
        t = t + E / la
        spike.append(t)

    #return np.array(spike)
    return spike