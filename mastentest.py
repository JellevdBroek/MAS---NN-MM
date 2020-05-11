import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Creates and trains Neural Network
def main():

    # Initialize data
    X = np.array(filetomatrixX("X_MAS.CSV"))
    Y = filetomatrixY("Y_MAS.CSV", 6)[5]
    bs = baseline(X, Y)
    
    # Preprocess data
    X = normdata(X)[0]
    X = X.reshape(-1, 50, 50, 1)
    trainX, trainY, testX, testY = splitdata(X, Y, 0.8)

    # Construct NN
    model = tf.keras.models.Sequential()

    model.add(Conv2D(50, (3,3), input_shape=trainX.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(100, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(100, (3,3)))
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(tf.keras.layers.Dense(100))
    model.add(Activation("relu"))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])

    # Trains neural network
    ne = 1
    history = model.fit(trainX, trainY, batch_size = 16, epochs=ne, validation_data=(testX, testY))

    # Shows performance
    plt.plot(history.history['mean_absolute_error'], label='mean_absolute_error')
    plt.plot(history.history['val_mean_absolute_error'], label = 'val_mean_absolute_error')
    plt.hlines(bs, 0, ne, colors='red')
    plt.xlabel('Epoch')
    plt.ylabel('mean_absolute_error')
    plt.legend(loc='lower right')
    plt.show()

    model.save('models/testmodel') 

# Splits the data in test and train set
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
    return np.array(trainx), np.array(trainy), np.array(testx), np.array(testy)

# Extracts X or B data from csv file
def filetomatrixX(file):
    matrixlist = []
    f = open(file)
    fl = list(f)
    for rown in range(len(fl)):
        if rown % 2 == 0:
            matrix = []
        newrow = fl[rown].split(",")
        for datan in range(len(newrow)):
            if datan % 50 == 0:
                matrixrow = []
            if newrow[datan][-1] == "\n":
                newrow[datan] = newrow[datan][:-1]
            matrixrow.append(float(newrow[datan]))
            if datan % 50 == 49:
                matrix.append(matrixrow)
        if rown % 2 == 1:
            matrixlist.append(matrix)
    f.close()
    return matrixlist

# Extracts Y data from csv file
def filetomatrixY(file, num):
    f = open(file)
    fl = list(f)
    matrix = [[] for _ in range(num)]
    for rown in range(len(fl)):
        newrow = fl[rown].split(",")
        if newrow[-1][-1] == "\n":
            newrow[-1] = newrow[-1][:-1]
        for data in newrow:
            matrix[rown % num].append(float(data))
    return matrix

# Normalizes data
def normdata(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    newdata = data - mean
    newdata /= std
    return newdata, mean, std

# Calculates what the MAE would be if algorithm would 
# predict the current payoff as future payoff
def baseline(X, Y):
    P = []
    for matrix in X:
        P.append(np.sum(matrix))
    M = []
    for n in range(len(X)):
        M.append(abs(float(P[n]) - float(Y[n])))
    return np.mean(M)

if __name__ == "__main__":
    main()



