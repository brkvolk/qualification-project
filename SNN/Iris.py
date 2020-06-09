#################################################################
#
#   SNN learning and classification on iris dataset
#
#################################################################
import pandas as pd
import numpy as np
from SNN.spike_gen import gen_spike
from SNN.network import NeuralNetwork
from SNN.weight_initialization import weight_init
from SNN.network_parameters import I, H, O, ny, la, T, L
import matplotlib.pyplot as plt
#
path = "C:\\Users\\ALEX\\PycharmProjects\\3-neuron_network\\SNN\\iris\\"
data = pd.read_csv(path + "iris_csv.csv")


sepallength = [min(data['sepallength']), max(data['sepallength'])]
sepalwidth = [min(data['sepalwidth']), max(data['sepalwidth'])]
petallength = [min(data['petallength']), max(data['petallength'])]
petalwidth = [min(data['petalwidth']), max(data['petalwidth'])]

# print()

#разбиваем по классам
setosa = data[data['class'] == u'Iris-setosa']
versicolor = data[data['class'] == u'Iris-versicolor']
virginica = data[data['class'] == u'Iris-virginica']

#разбиваем по датасетам
setosa_learn, setosa_test = np.array_split(setosa, 2)
versicolor_learn, versicolor_test = np.array_split(versicolor, 2)
virginica_learn, virginica_test = np.array_split(virginica, 2)

test = setosa_test.append(versicolor_test).append(virginica_test)
learn = setosa_learn.append(versicolor_learn).append(virginica_learn)
#ставим индексы по порядку
test.index = np.arange(len(test))
learn.index = np.arange(len(learn))

# print(learn)
# print(test.loc[28]['sepallength'])
learn = learn.sample(frac=1).reset_index(drop=True)#перемешиваем строки
# print(learn)
#целевые спайки
setosa_target = gen_spike(16)
versicolor_target = gen_spike(18)
virginica_target = gen_spike(20)
# setosa_target = gen_spike(32)
# versicolor_target = gen_spike(36)
# virginica_target = gen_spike(40)

targets=[]
for i in range(75):
    if learn.loc[i]['class'] == 'Iris-setosa': targets.append(setosa_target)
    elif learn.loc[i]['class'] == 'Iris-versicolor': targets.append(versicolor_target)
    elif learn.loc[i]['class'] == 'Iris-virginica': targets.append(virginica_target)


# кодируем в спайки
freq_range = [15, 30]
learning_data_row = []
testing_data_row = []

for i in range(75):
    spike1 = gen_spike(np.interp(test.loc[i]['sepallength'], sepallength, freq_range))
    spike2 = gen_spike(np.interp(test.loc[i]['sepalwidth'], sepalwidth, freq_range))
    spike3 = gen_spike(np.interp(test.loc[i]['petallength'], petallength, freq_range))
    spike4 = gen_spike(np.interp(test.loc[i]['petalwidth'], petalwidth, freq_range))
    testing_data_row.append([spike1, spike2, spike3, spike4])

    spike1 = gen_spike(np.interp(learn.loc[i]['sepallength'], sepallength, freq_range))
    spike2 = gen_spike(np.interp(learn.loc[i]['sepalwidth'], sepalwidth, freq_range))
    spike3 = gen_spike(np.interp(learn.loc[i]['petallength'], petallength, freq_range))
    spike4 = gen_spike(np.interp(learn.loc[i]['petalwidth'], petalwidth, freq_range))
    learning_data_row.append([spike1, spike2, spike3, spike4])

# обучение
# Wih, Who = np.load(path+'Wih.npy'), np.load(path+'Who.npy')
Wih, Who = weight_init()
setosa_error = []
versicolor_error = []
virginica_error = []
# print(Who)
epoch = 0
max_epoch = 100
NeuralNetwork = NeuralNetwork([], [], Wih, Who)
# print(NeuralNetwork.weights2)
k = np.zeros(100)
while epoch < max_epoch:
    for i in range(75):
        NeuralNetwork.input = learning_data_row[i]
        NeuralNetwork.target = targets[i]

        NeuralNetwork.quantization()
        NeuralNetwork.feedforward()
        NeuralNetwork.ERROR()
        # print(NeuralNetwork.weights2)
        if NeuralNetwork.error >= 0.001:
            k[epoch] += 1

        if learn.loc[i]['class'] == 'Iris-setosa':
            setosa_error.append(NeuralNetwork.error)
        elif learn.loc[i]['class'] == 'Iris-versicolor':
            versicolor_error.append(NeuralNetwork.error)
        elif learn.loc[i]['class'] == 'Iris-virginica':
            virginica_error.append(NeuralNetwork.error)

        NeuralNetwork.backprop()
        # print(NeuralNetwork.weights2)
        # Wih, Who = NeuralNetwork.weights1, NeuralNetwork.weights2
        # print(Wih,"\n")
        # print(Who,"\n+++++++++++++++++++++")
        print(i)
        print("output = ", NeuralNetwork.output)
        print("target =", NeuralNetwork.target)
        print('ERROR:', NeuralNetwork.error)
        print("epoch:", epoch, "\nmissclassified:", k[epoch])
        print("=================================================================================================================================================================")
    epoch += 1

np.save(path+'Wih2.npy', NeuralNetwork.weights1)
np.save(path+'Who2.npy', NeuralNetwork.weights2)
np.save(path+'setosa_error.npy', np.array(setosa_error))
np.save(path+'versicolor_error.npy', np.array(versicolor_error))
np.save(path+'virginica_error.npy', np.array(virginica_error))

print("learning complited")
print(k)
#классификация
Wih, Who = np.load(path+'Wih2.npy'), np.load(path+'Who2.npy')
# print(Wih)

setosa_error = []
versicolor_error = []
virginica_error = []

NeuralNetwork1 = NeuralNetwork([], [], Wih, Who)
for i in range(75):
    NeuralNetwork1.input = testing_data_row[i]
    NeuralNetwork1.feedforward()
    if test.loc[i]['class'] == 'Iris-setosa':
        setosa_error.append(NeuralNetwork1.error)
    elif test.loc[i]['class'] == 'Iris-versicolor':
        versicolor_error.append(NeuralNetwork1.error)
    elif test.loc[i]['class'] == 'Iris-virginica':
        virginica_error.append(NeuralNetwork1.error)
np.save(path+'setosa_error_test.npy', np.array(setosa_error))
np.save(path+'versicolor_error_test.npy', np.array(versicolor_error))
np.save(path+'virginica_error_test.npy', np.array(virginica_error))

print(setosa_error)
print(versicolor_error)
print(virginica_error)






