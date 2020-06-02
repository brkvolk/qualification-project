
import math

def timeIP(tm , tn):#скалярное произведение моментов времени
    sigma = 2.0
    return math.exp( - (abs(tm - tn) ** 2) / (2 * (sigma ** 2) ) )

def spikeIP(spike1, spike2):#скалярное произведение спайков
   S = 0
   for tm in spike1:
       for tn in spike2:
           S += timeIP(tm, tn)
   #print(S)
   return S