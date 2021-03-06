#################################################################
#
#   SNN learning and classification on iris dataset
#
#################################################################
import pandas as pd
import numpy as np
from SNN.spike_gen import gen_spike
from SNN.inner_products import spikeIP
from SNN.network import NeuralNetwork
from SNN.weight_initialization import weight_init
from SNN.network_parameters import I, H, O, ny, la, T, L
import matplotlib.pyplot as plt

def Error(output,target):
    return(0.5 * (spikeIP(output, output) - 2 * (spikeIP(output, target)) + spikeIP(target,target)))


#
path = "C:\\Users\\ALEX\\PycharmProjects\\3-neuron_network\\SNN\\iris\\"

# sepallength = [min(data['sepallength']), max(data['sepallength'])]
# sepalwidth = [min(data['sepalwidth']), max(data['sepalwidth'])]
# petallength = [min(data['petallength']), max(data['petallength'])]
# petalwidth = [min(data['petalwidth']), max(data['petalwidth'])]

# print()

#разбиваем по классам
# setosa = data[data['class'] == u'Iris-setosa']
# versicolor = data[data['class'] == u'Iris-versicolor']
# virginica = data[data['class'] == u'Iris-virginica']
#
# #разбиваем по датасетам
# setosa_learn, setosa_test = np.array_split(setosa, 2)
# versicolor_learn, versicolor_test = np.array_split(versicolor, 2)
# virginica_learn, virginica_test = np.array_split(virginica, 2)

# test = setosa_test.append(versicolor_test).append(virginica_test)
# learn = setosa_learn.append(versicolor_learn).append(virginica_learn)
#ставим индексы по порядку
# test.index = np.arange(len(test))
# learn.index = np.arange(len(learn))

# print(learn)
# print(test.loc[28]['sepallength'])
# learn = learn.sample(frac=1).reset_index(drop=True)#перемешиваем строки
# learn.to_csv(path + "learn.csv")
data = pd.read_csv(path + "iris_csv.csv")
# learn = pd.read_csv(path + "please\\new.csv")
learn = pd.read_csv(path + "learn.csv")
# learn = pd.read_csv(path + "iris_csv.csv")
# learn.index = np.arange(len(learn))
learn = learn.sample(frac=1).reset_index(drop=True)#перемешиваем строки
# print(learn)
sepallength = [min(data['sepallength']), max(data['sepallength'])]
sepalwidth = [min(data['sepalwidth']), max(data['sepalwidth'])]
petallength = [min(data['petallength']), max(data['petallength'])]
petalwidth = [min(data['petalwidth']), max(data['petalwidth'])]

# кодируем в спайки
# freq_range = [30, 45]
freq_range = [9, 16]
learning_data_row = []
# testing_data_row = []

for i in range(75):
#     spike1 = gen_spike(np.interp(test.loc[i]['sepallength'], sepallength, freq_range))
#     spike2 = gen_spike(np.interp(test.loc[i]['sepalwidth'], sepalwidth, freq_range))
#     spike3 = gen_spike(np.interp(test.loc[i]['petallength'], petallength, freq_range))
#     spike4 = gen_spike(np.interp(test.loc[i]['petalwidth'], petalwidth, freq_range))
#     testing_data_row.append([spike1, spike2, spike3, spike4])

    spike1 = gen_spike(np.interp(learn.loc[i]['sepallength'], sepallength, freq_range))
    spike2 = gen_spike(np.interp(learn.loc[i]['sepalwidth'], sepalwidth, freq_range))
    spike3 = gen_spike(np.interp(learn.loc[i]['petallength'], petallength, freq_range))
    spike4 = gen_spike(np.interp(learn.loc[i]['petalwidth'], petalwidth, freq_range))
    learning_data_row.append([spike1, spike2, spike3, spike4])

# print(learning_data_row)


#целевые спайки
# setosa_target = gen_spike(14)
# versicolor_target = gen_spike(16)
# virginica_target = gen_spike(18)


# setosa_target = gen_spike(32)
# versicolor_target = gen_spike(36)
# virginica_target = gen_spike(40)
# setosa_target, versicolor_target, virginica_target = np.load(path+'setosa_target.npy'), np.load(path+'versicolor_target.npy'), np.load(path+'virginica_target.npy')
setosa_target, versicolor_target, virginica_target = np.load(path+'setosa_target.npy'), np.load(path+'versicolor_target.npy'), np.load(path+'virginica_target.npy')
np.array(setosa_target).tolist()
np.array(versicolor_target).tolist()
np.array(virginica_target).tolist()

print("setosa-virginica", spikeIP(setosa_target, virginica_target))
print("setosa-versicolor", spikeIP(setosa_target, versicolor_target))
print("versicolor-virginica", spikeIP(versicolor_target, virginica_target))

targets = []
for i in range(75):
    if learn.loc[i]['class'] == 'Iris-setosa': targets.append(setosa_target)
    elif learn.loc[i]['class'] == 'Iris-versicolor': targets.append(versicolor_target)
    elif learn.loc[i]['class'] == 'Iris-virginica': targets.append(virginica_target)

# np.save(path+'setosa_target.npy', np.array(setosa_target))
# np.save(path+'versicolor_target.npy', np.array(versicolor_target))
# np.save(path+'virginica_target.npy', np.array(virginica_target))

# обучение
# Wih, Who = np.load(path+'please\\Wih_buff.npy'), np.load(path+'please\\Who_buff.npy')
Wih, Who = np.load(path+'please\\Wih.npy'), np.load(path+'please\\Who.npy')
# Wih, Who = np.load(path+'please\\Wih12.npy'), np.load(path+'please\\Who12.npy')
# Wih, Who = weight_init()
# np.save(path+'please\\Wih0.npy', Wih)
# np.save(path+'please\\Who0.npy', Who)

# print(Wih, Who)
epoch = 0
max_epoch = 20

NeuralNetwork = NeuralNetwork([], [], Wih, Who)
# NeuralNetwork.quantization()
k = np.zeros(max_epoch)

setosa_error = np.zeros(max_epoch)
versicolor_error = np.zeros(max_epoch)
virginica_error = np.zeros(max_epoch)


while epoch < max_epoch:

    for i in range(75):

            NeuralNetwork.input = learning_data_row[i]
            NeuralNetwork.target = targets[i]

            NeuralNetwork.feedforward()
            NeuralNetwork.ERROR()




        # Error[epoch] += NeuralNetwork.error
            E = min(Error(NeuralNetwork.output, setosa_target), Error(NeuralNetwork.output, versicolor_target), Error(NeuralNetwork.output, virginica_target))

            # print(Error(NeuralNetwork.output, setosa_target))
            # print(Error(NeuralNetwork.output, versicolor_target))
            # print(Error(NeuralNetwork.output, virginica_target))
            # print("E = ", E)
            # print("network error", NeuralNetwork.error )

            if (NeuralNetwork.error == E) & (len(NeuralNetwork.output) != 0):

                k[epoch] += 1

                if learn.loc[i]['class'] =='Iris-setosa': setosa_error[epoch] += 1
                elif learn.loc[i]['class'] == 'Iris-versicolor': versicolor_error[epoch] += 1
                elif learn.loc[i]['class'] == 'Iris-virginica': virginica_error[epoch] += 1
            # elif (NeuralNetwork.error != E) & (len(NeuralNetwork.output) != 0):

            NeuralNetwork.backprop()
            print(i)
            print("output = ", NeuralNetwork.output, "\nlen:", len(NeuralNetwork.output))
            print("target =", NeuralNetwork.target, "\nlen:", len(NeuralNetwork.target))
            print('ERROR:', NeuralNetwork.error)
            print("epoch:", epoch, "\n correctly classified:", k[epoch])
            print("=================================================================================================================================================================")

    epoch +=1

    print(k)

print (setosa_error)
print (versicolor_error)
print (virginica_error)



x = [i for i in range(epoch)]
    # y3 = [Error[i] for i in x3]
    # plt.plot(x3, y3)
    # plt.show()
y1 = [25-setosa_error[i] for i in x]
y2 = [25-versicolor_error[i] for i in x]
y3 = [25-virginica_error[i] for i in x]
y4 = (setosa_error+virginica_error+versicolor_error)/0.75

plt.plot(x, y1, color='blue')
plt.plot(x, y2, color='red')
plt.plot(x, y3, color='green')
plt.show()

# plt.plot(x, y4, color='black')
# plt.show()



np.save(path+'please\\Wih.npy', NeuralNetwork.weights1)
np.save(path+'please\\Who.npy', NeuralNetwork.weights2)
np.save(path+'please\\setosa_error.npy', np.array(setosa_error))
np.save(path+'please\\versicolor_error.npy', np.array(versicolor_error))
np.save(path+'please\\virginica_error.npy', np.array(virginica_error))

print("learning complited")
print(k)
# x3 = [i for i in range(max_epoch)]
# y3 = [k[i] for i in x3]
# plt.plot(x3, y3)
# plt.show()





