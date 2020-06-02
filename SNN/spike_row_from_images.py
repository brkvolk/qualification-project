########################################################################################################################
#                                                                                                                      #
#           imgs(white background) are binarizing & encoding into spike trains                                          #
#           with distanse(=refractory period) tau between spikes                                                        #
#                                                                                                                      #
########################################################################################################################
import numpy as np
import cv2


def binarization_Bradley(img):#for low contrast imgs

    weight = img.shape[0]
    height = img.shape[1]
    s = weight//16 #window size s x s
    bright_th = 0.15

    #src = [k for k in img] #vector from img
    src = np.ravel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))# img can be colored
    res = np.zeros(weight*height)
    integral_image = np.zeros(weight*height)

    for i in range(weight):
        sum = 0
        for j in range(height):
            index = j * weight + i
            sum += src[index]
            if (i == 0):
                integral_image[index] = sum
            else:
                integral_image[index] = integral_image[index - 1] + sum

    # находим границы для локальные областей
    for i in range(weight):
        for j in range(height):
            index = j * weight + i
            x1 = i - s//2
            x2 = i + s//2
            y1 = j - s//2
            y2 = j + s//2

            # border check
            if x1 < 0 :
                x1 = 0
            if x2 >= weight:
                x2 = weight - 1
            if y1 < 0:
                y1 = 0
            if y2 >= height:
                y2 = height - 1

            count = (x2-x1) * (y2-y1)

            sum = integral_image[y2 * weight + x2] - integral_image[y1 * weight + x2] - integral_image[y2 * weight + x1] + integral_image[y1 * weight + x1]
            if (src[index] * count < sum * (1.0 - bright_th)):
                res[index] = 0#black
            else:
                res[index] = 255#white

    res = np.resize(res, (weight, height))
    return (res)

def img_to_spikes(matrix, weight, height):
    tau = 0.1/weight
    spikes = []
    for i in range(weight):
        spikes.append([])

        for j in range(height):
            if (matrix[i][j] == 0) :
                spikes[i].append(tau*j)


    return spikes


img = cv2.imread(r"C:\Users\ALEX\PycharmProjects\3-neuron_network\SNN\imgs\text.jpg")
cv2.imshow("img", img)

img2 = binarization_Bradley(img)
ret, img3 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow("img2", img2)
cv2.imshow("img3", img3)
print(img2)
spikes = img_to_spikes(img2, img.shape[0], img.shape[1])
print (spikes[(img.shape[1])//2])

cv2.waitKey(0)
