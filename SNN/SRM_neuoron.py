##############################################################
# Computing potential of neuron from all input spike trains
# at current time t.
# SRM0 neuron  model was used
##############################################################

import numpy as np
import math
from SNN.network_parameters import T, I

P_rest = 0.
P_threshold = 1 # это должно быть то же число, что и в ф-ции эта?
delta_t = 0.0001

class neuron:
	def __init__(self):
		self.P_th:	 float = P_threshold
		self.P_rest: float = P_rest
		self.t_rest: float = 0.
		self.P:		 float = P_rest
		self.dt: 	 float = delta_t
		self.output:  list = []
		self.data:	  list = [] # значение потенциала в момент t+dt

	# потенциал одного нейрона
	def potential(self, spikes, vect_of_w):

		self.output = []
		self.data = []
		t_hat = -math.inf     #выпускаемый спайк
		t = 0
		t_ref 	= 0 #время, с которого опять учитываем новые спайки
		t_firing_last = np.zeros(len(vect_of_w)) #вспомогательный массив последних спайков

		while t <= T:
			k = 0         # номер синапса/входного спайка
			for spike in spikes:          #идем по входным спайкам
				spike_sum = 0             #и суммируем вклад от каждого в момент времени t

				for t_firing in spike:                                                          #проверяем выпустился ли новый спайк
					if (t_firing <= t) & (t_firing > t_firing_last[k]) & (t >= t_ref):			#и нужно лии его учитывать
						t_firing_last[k] = t_firing

				spike_sum += eps_spike_response(t - t_firing_last[k]) * vect_of_w[k]     		# прибавляем вклад спайка к общей сумме
																								#на предидущем шаге учли, какой именно спайк вносит вклад
				k += 1

			self.P = eta_refract(t - t_hat) + spike_sum                 #Потенциал в момент t

			if self.P - self.P_th >= 0 :                       #проверяем,  пересекли ли порог
				t_ref = t + tR								   #если да, то некоторое время не считаем новые одиночные спайки
				t_hat = t									   # и выпускаем с нейрона выходной одиночный спайк
				self.output.append(t_hat)

				#else :#не учитываем спайки
				#	spike_sum += eps_spike_response(t - t_firing_last[k])* vect_of_w[k]
				#	self.P = eta_refract(t - t_hat) + spike_sum

			#записываем время и потенциал, чтобы построить график
			self.data.append(self.P)
			t += self.dt


	def initial(self):
		self.t_rest = 0
		self.P = P_rest


#SRM
tau = 0.005
tauR = 0.05
threshold = 1 #порог
tR = 0.001 # мин рассстояние между 2 спайками

def eta_refract(t):
	if t > 0.:
		return - threshold * math.exp(- t / tauR)
	else:
		return 0

def eps_spike_response(t):
	if t > 0.:
		return t*math.exp(1 - (t / tau)) / tau
	else:
		return 0
