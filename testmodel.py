import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import MASdata as md
import mastentest as mt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from collections import defaultdict

# Plots histograns of future states based on board states.
def main():
    # Get index for board states
    N = [random.randint(100, 24000) for _ in range(10)]

    # Gets data
    X = mt.filetomatrixX("X_MAS.csv")
    PX = [X[n] for n in N]
    X = np.array(X).reshape(-1, 50, 50, 1)  
    B = mt.filetomatrixX("B_MAS.csv")
    X = mt.normdata(X)[0]
    Y = mt.filetomatrixY("Y_MAS.CSV", 6)[5]

    # Load in neural network
    new_model = tf.keras.models.load_model('models/test_10')

    d = 0

    # Makes one histogram
    for m in range(len(N)):
        n = N[m]
        print(d)
        d += 1
        name = "models/plot" + str(n) + "test"
        sv = np.sum(PX[m])
        NX = X[n]
        NX = np.array(NX).reshape(-1, 50, 50, 1)  
        a = new_model.predict(NX)[0][0]
        pv = btopd(B[n], 2500, 250, 10, r=0, gamma=0)
        plthis(pv, 10, name, val=a, tv=Y[n], sv=sv)

# Gets distribution of total payoffs for x time steps
def btopd(board, ua, n, timesteps, r=0, gamma=0):
    pblist = []
    if r == False:
        r = 2.8
    if gamma == False:
        gamma = 0.35
    pb = md.payboard(board, r, gamma)


    for m in range(n):
        npb = pb.copy()
        nboard = board.copy()
        if m % 100 == 0:
            print(m)

        # Runs 1 simulation with x time steps
        for _ in range(timesteps):
            nboard, npb = md.nextboard(nboard, npb, ua, r, gamma, 0.5, show=False)
        pblist.append(np.sum(npb))
    return pblist

# Plots histogram with vector based on interval
def plthis(vec, inter, name, val="N", tv="N", sv="N"):

    valdic = {} 
    valdic = defaultdict(lambda:0,valdic)
    for n in vec:
        valdic[int(n/inter) * inter] += 1
    a = list(valdic.values())
    b = list(valdic.keys())
    ma = np.max(a)
    plt.figure()
    plt.bar(b, a, width=inter)
    if val != "N":
        plt.axvline(val, 0, ma, label='predicted value', color="black")
    if tv != "N":
        plt.axvline(tv, 0, ma, label='True Y value', color="red")
    if sv != "N":
        plt.axvline(sv, 0, ma, label='Total start value', color="green")
    plt.savefig(name)
    return valdic

if __name__ == "__main__":
    main()
