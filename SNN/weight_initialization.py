###########################################################################
#
# Weights initialisation + rules
#
###############################################################


import numpy as np
from SNN.network_parameters import I, H, O, ny


def weight_init():
    Wih = np.random.uniform(0, 0.3, (I, H)) #матрицы весов
    if O == 1:
        Who = np.random.uniform(0., 0.2, H)
    else:
        Who = np.random.uniform(0, 0.2, (H, O))
    return Wih, Who

#правила для весов

from SNN.inner_products import spikeIP

def dwoh( spike_d, spike_a, spike_h):
    return -ny*(spikeIP(spike_a, spike_h) - spikeIP(spike_d, spike_h))

def dwhi( spike_d, spike_a, spike_i, who):
     return -ny*(spikeIP(spike_a, spike_i) - spikeIP(spike_d, spike_i))*who
