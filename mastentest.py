import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

def splitdata(x, y, ratio):
    trainx = []
    trainy = []
    testx = []
    testy= []
    for n in range(len(x)):
        if random.random() < ratio:
            trainx.append(x[n])
            trainy.append(y[n])
        else:
            testx.append(x[n])
            testy.append(y[n])
    return trainx, trainy, testx, testy

def filetomatrix(file, rows=1):
    matrix = []
    f = open(file)
    fl = list(f)
    for rown in range(len(fl)):
        newrow = fl[rown].split(",")
        if newrow[-1][:-2] == "\n":
            newrow[-1] = newrow[-1][:-2]
        for n in range(len(newrow)):
            newrow[n] = float(newrow[n])
        if rown % rows == 0:
            print(newrow)
            matrix.append(newrow)
        else:
            matrix[-1] += newrow
    f.close()
    return matrix

def mtoms(data):
    ndata = []
    for m in data:
        nm = m.reshape(50, 50)
        ndata.append(nm)
    return ndata

def normdata(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    newdata = data - mean
    newdata /= std
    return newdata, mean, std

X = np.array(filetomatrix("TXMAS.CSV", rows=2))
print(X[0][0])
X = np.array(mtoms(X))
print(X[0][0])
Y = np.array(filetomatrix("TYMAS.CSV"))

NX, mx, sx = normdata(X)
NY, my, sy = normdata(Y)

trainX, trainY, testX, testY = splitdata(NX, Y, 0.8)

TRX = np.array(trainX)
TEX = np.array(testX)
TRY = np.array(trainY)
TEY = np.array(testY)

print(TRY.shape)
print(TRX.shape)

# model = tf.keras.models.Sequential()
# model.add( Conv2D(64, (5,5), input_shape=(50, 50, 1)))
# model.add(Activation("relu"))

# model.add(Flatten())
# model.add(tf.keras.layers.Dense(1))

# model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])
# model.fit(TRX, TRY, batch_size = 16, epochs = 3, validation_split=0.1)

# aa = model.evaluate(TEX, TEY)
# print(aa)