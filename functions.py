import numpy as np
import random
import math


T = 0.1 #time period
la = 20.0 #frequencie 
ny = 0.005 #backprop learning parametr

Wih = np.random.uniform(0, 0.2, (2, 2)) #matrixes of waights of input->hiden, hiden->output layers
Who = np.random.uniform(0, 0.2, 2)

#rules of waight`s change
def dwoh( spike_d, spike_a, spike_h):
    return -ny*(spikeIP(spike_a, spike_h)-spikeIP(spike_d, spike_h))

def dwhi( spike_d, spike_a,spike_i,who):
     return -ny*(spikeIP(spike_a, spike_i)-spikeIP(spike_d, spike_i))*who

def Error( spike_d, spike_a):
    return 0.5*(spikeIP(spike_a, spike_a)-2*spikeIP(spike_a, spike_d)+spikeIP(spike_d, spike_d))

def gen_spike(): #generate spikes using Poisson process
    spike = []
    t = 0
    k = 0
    while t < T:
        E = random.expovariate(la)
        k = k + 1
        t = t + E / la
        spike.append(t)
    return spike, k

def timeIP(tm , tn):#inner product of spike times
    sigma=2.0
    return math.exp(-(abs(tm-tn)**2)/(2*sigma))

def spikeIP(spike1, spike2):#сinner product of spike series
   S=0
   for tm in spike1:
       for tn in spike2:
           S += timeIP(tm, tn)
   #print(S)
   return S

def spike_sum (spike1, spike2):
     spike_merge = spike1
     spike_merge.extend(spike2)
     spike_merge.sort()
     return spike_merge

def multipl(spike, k):
    spike = [i * k for i in spike if i*k <= 0.1]
    return spike

def sigmoid(x):
  return (1 / (1 + math.exp(-x)))

#SRM neuron model
tau = 0.005#constant of postsynaptic potential
tauR = 0.05#refractory period
nu = 1 #neuron threshold
tR = 0.001 # absolut refractory period
def eta_refract(t):
    if t > 0.:
        return -nu*math.exp(-t/tauR)
    else:
        return 0
def spike_response(t):
    if t > 0.:
        return t*math.exp(1-(t/tau))/tau
    else:
        return 0

def closest_t(t, spike):#nearest to the left of t single spike
    с = 1
    for i in spike :
        if t - i > 0 :
            c = i
        else :
            break
    return c

def neuron_state(t, spikes, vect_of_w):#how to consider tR????
    S = 0
    k = 0
    tf = 1
    for spike in spikes :
        spike_sum = 0
        tf_loc = closest_t(t, spike)
        for i in spike:
             spike_sum += spike_response(t - i)
        S = S + spike_sum * vect_of_w[k]
        k += 1
        if tf == 1 :
            tf = tf_loc
        elif (tf < tf_loc) & (tf_loc < t) :
            tf = tf_loc

    return eta_refract(t - tf) + S


#just check if i am screwed up

max_epoch=500

spike=[]
spike1, m = gen_spike()
spike2, n = gen_spike()
d_spike, k = gen_spike()
spike_s =spike_sum (spike1, spike2)

print("spike1", spike1)
print ("spike1 length", m)
print(spike2)
print(len(spike2))
#print("d_spike:", d_spike)
#print("spike_sum", spike_s )
#print(len(spike_s ))
print(spikeIP(spike1, spike2))
print(spikeIP(spike1, spike1))


tf=max(spike1)

