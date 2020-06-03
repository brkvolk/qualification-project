##############################################################
# Computing potential of neuron from all input spike trains
# at current time t.
# SRM0 neuron  model was used
##############################################################
import copy
import numpy as np
import math
from SNN.network_parameters import T, delta_t, P_rest, P_threshold, tau, tauR, tR, threshold

class neuron:
	def __init__(self):
		self.P_threshold: float = P_threshold
		self.P_rest: float = P_rest
		self.t_rest: float = 0.
		self.P: float = P_rest
		self.dt: float = delta_t
		self.output: list = []
		self.data: list = [] # значение потенциала в момент t+dt

	# потенциал одного нейрона
	def potential(self, spikes, vect_of_w):
		# print(spikes)
		self.output = []
		self.data = []
		t_hat = -math.inf     #выпускаемый спайк
		t = 0
		t_ref = 0 #время, с которого опять учитываем новые спайки
		t_firing_last = np.zeros(len(vect_of_w)) #вспомогательный массив последних спайков

		current_spikes = copy.deepcopy(spikes)

		while t <= T:
			#print(current_spikes)
			k = 0         # номер синапса/входного спайка
			buff_spikes = []
			for spike in current_spikes:          #идем по входным спайкам
				spike_sum = 0             #и суммируем вклад от каждого в момент времени t

				# for t_firing in spike:                                                          #проверяем выпустился ли новый спайк
				# 	if (t_firing <= t) & (t_firing > t_firing_last[k]) & (t >= t_ref):			#и нужно ли его учитывать
				# 		t_firing_last[k] = t_firing

				# if spike:
				# 	if (spike[0] <= t):
				# 		buff_t = spike.pop(0)
				# 		if (t >= t_ref):
				# 			t_firing_last[k] = buff_t
				# 		buff_spikes.append(spike)
				# 	else:
				# 		buff_spikes.append(spike)
				# else:
				# 	buff_spikes.append(spike)
				if spike:
					if (spike[0] <= t):
						buff_t = spike.pop(0)
						if (t >= t_ref):
							t_firing_last[k] = buff_t

				buff_spikes.append(spike)

				spike_sum += eps_spike_response(t - t_firing_last[k]) * vect_of_w[k]     		# прибавляем вклад спайка к общей сумме
				k += 1																		#на предидущем шаге учли, какой именно спайк вносит вклад

			current_spikes = buff_spikes.copy()

			# esum = 0
			# for m in self.output:
			# 	esum += eta_refract(t - m)
			#
			esum = eta_refract(t - t_hat)

			self.P = esum + spike_sum                                                  #Потенциал в момент t


			if ((self.P - self.P_threshold) >= 0) :  # проверяем,  пересекли ли порог
				t_ref = t + tR								   								#если да, то некоторое время не считаем новые одиночные спайки
				t_hat = t									  								# и выпускаем с нейрона выходной одиночный спайк
				self.output.append(t_hat)

			self.data.append(self.P)
			t += self.dt
		# print(spikes,"\n")
			# if self.data:
			# 	P_prev = self.data[-1]
			# else: P_prev = 0
			#
			# if ((self.P - self.P_th) >= 0) & (P_prev - self.P < 0):                        #проверяем,  пересекли ли порог
			# 	t_ref = t + tR								   								#если да, то некоторое время не считаем новые одиночные спайки
			# 	t_hat = t									  								# и выпускаем с нейрона выходной одиночный спайк
			# 	self.output.append(t_hat)


	def initial(self):
		self.t_rest = 0
		self.P = self.P_rest
		self.data = []


#SRM

def eta_refract(t):
	if t > 0:
		return (- threshold * math.exp(- t / tauR))
	else:
		return (0)

def eps_spike_response(t):
	if t > 0.:
		return ( (t * math.exp(1 - (t / tau)) )/ tau)
	else:
		return (0)
