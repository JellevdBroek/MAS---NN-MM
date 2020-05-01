import numpy as np
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from matplotlib.colors import LinearSegmentedColormap
import time
import random

# Strategy index
# 0 = C
# 1 = D
# 2 = L
# 3 = M0
# 4 = M1
# 5 = M2
# 6 = M3
# 7 = M4

def setboard(N, strats):
    # Distrubutes given strategies randomly
    m = len(strats)-1
    board = []
    for _ in range(N):
        row = []
        for _ in range(N):
            row.append(strats[random.randint(0, m)])
        board.append(row)
    return np.array(board)

def paygroup(strat, center, board, r, gamma):
    # Calculates payoff if a certain strategy was in a group
    lb = len(board[0])
    if  strat == 2:
        return 1
    
    ay = center[0]
    ax = center[1]
    c, d, m = 0,0,0

    # The agents around the center of the group
    group = [board[ay][ax], board[(ay-1)% lb][ax], board[(ay+1)% lb][ax], board[ay][(ax-1)% lb], board[ay][(ax+1)% lb]]
    
    for member in group:
        if member == 0:
            c += 1
        elif member == 1:
            d += 1
            
    for member in group:
        if member > 2 and d < member - 3:
            m += 1
    
    if strat > 2 and d >= strat - 3:
        return 1 - gamma
    
    po = r * (c + m) / (d + c + m)
    
    if strat == 1:
        return po
    
    elif strat == 0:
        return po - 1
    
    else:
        return po - 1 - gamma

def payoff(ax, ay, board, r, gamma):
    # Calculates the total payoff of an agent
    lb = len(board[0])
    strat = board[ay][ax]
    po = 0
    
    # Takes center of all 5 groups
    cords = [(ay,ax), ((ay-1)%lb,ax), ((ay+1)%lb,ax), (ay,(ax-1)%lb), (ay,(ax+1)%lb)]
    for cord in cords:
        po += paygroup(strat, cord, board, r, gamma)
    return po

def payboard(board, r, gamma):
    # Calculates the payoff for every agent and creates heatmap
    N = len(board[0])
    pb = []
    for ay in range(N):
        for ax in range(N):
            pb.append(payoff(ax, ay, board, r, gamma))
    pb = np.reshape(pb, (N, N))
    return pb


def transfunc(a1x, a1y, a2x, a2y, board, pb, K):
    # Calculates the probabilty of agent 1 changing strategy to agent 2s strategy
    lb = len(board[0])
    payoff1 = pb[a1y][a1x]
    payoff2 = pb[a2y][a2x]
    prob = 1/(1 + np.exp(payoff1 - payoff2)/K)

    # Returns new stragety for agent
    if random.random() > prob:
        return board[a1y][a1x]
    else:
        return board[a2y][a2x]
    
def adjagent(ax, ay, lb):
    # Selects a random adjecent agent
    nb = random.randint(0, 3)
    if nb == 0:
        return ax, (ay - 1) % lb
    if nb == 1:
        return (ax - 1) % lb, ay
    if nb == 2:
        return (ax + 1) % lb, ay
    else:
        return ax, (ay + 1) % lb
    
    
def countstrats(board, strats):
    # Count the occurence of all strategies
    stratdic = {}
    for n in strats:
        stratdic[n] = 0
    for row in board:
        for agent in row:
            stratdic[agent] += 1
    return stratdic
    
def updateboard(board, pb, ua, K):
    # 
    N = len(board[0])
    alist = np.random.choice(N**2, ua, replace=False)
    updatelist = []
    for agent in alist:
        a1y = int(agent/N)
        a1x = agent % N
        a2x, a2y = adjagent(a1x, a1y, N)
        updatelist.append(transfunc(a1x, a1y, a2x, a2y, board, pb, K))
    for m in range(ua):
        agent = alist[m]
        ay = int(agent/N)
        ax = agent % N
        board[ay][ax] = updatelist[m]
    return board

def plotshow(board, pb:
    # Shows the strategies on the board
    plt.imshow(board, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.show()
    # Shows the heatmap
    plt.imshow(pb, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.show()
    print(np.sum(pb))
    
def initboard(N, strats, r, gamma, show=True):
    # initializes the board
    board = setboard(N, strats)
    pb = payboard(board, r, gamma)
    if show:
        plotshow(board, pb)
    return board, pb
    
def nextboard(board, pb, ua, r, gamma, K, show=True):
    # Goes from the current board state to the next board state
    board = updateboard(board, pb, ua, K)
    pb = payboard(board, r, gamma)
    if show:
        plotshow(board, pb)
    return board, pb

def calcprop(board, strats, cp):
    # Keeps track of the propotions of strategies
    lb = len(board[0])
    sd = countstrats(board, strats)
    for n in range(len(strats)):
        cp[n].append(sd[strats[n]]/(lb ** 2))
    return cp

def simulation(N, strats, ua, r, gamma, K, timesteps, interval=1, showinterval=0):
    # Starts the simulation
    start_time = time.time()
    board, pb = initboard(N, strats, r, gamma, show=False)
    if showinterval:
        plotshow(board, pb, strats)
        print(countstrats(board, strats))
    cp = [[] for _ in range(len(strats))]
    cp = calcprop(board, strats, cp)
    timel = [0]
    pay = [np.sum(pb)]
    for n in range(1, timesteps+1):
        if n % 100 == 0:
            print(n)
        board, pb = nextboard(board, pb, ua, r, gamma, K, show=False)
        if n%interval == 0:
            timel.append(n)
            pay.append(np.sum(pb))
            cp = calcprop(board, strats, cp)
        if showinterval:
            if n%showinterval == 0 and n != timesteps:
                plotshow(board, pb, strats)
                print(countstrats(board, strats))
                time.sleep(1)
    plotshow(board, pb, strats)
    plt.figure()
    plt.plot(timel, pay)
    plt.show()
    plt.figure()
    for p in cp:
        plt.plot(timel, p)
    plt.show()
    print(countstrats(board, strats))
    print("--- %s seconds ---" % (time.time() - start_time))
    return pay, timel, board, pb

def hflatten(a):
    # splts the board in two equal sizes
    b = []
    la = len(a)
    a1 = np.array(a[:int(la/2)])
    a2 = np.array(a[int(la/2):])
    b.append(a1.flatten())
    b.append(a2.flatten())
    return b

def writetocsv(data, file):
    # saves data to csv
    f = open(file, "a")
    for row in data:
        f.write(str(round(row[0], 2)))
        for dp in row[1:]:
            f.write("," + str(round(dp, 2)))
        f.write("\n")
    f.close()

def datasimulation(N, strats, ua, r, gamma, K, timesteps, M, di):
    # simulation function without plots 
    board, pb = initboard(N, strats, r, gamma, show=False)
    x = []
    y = []
    ul = timesteps - di
    for n in range(1, timesteps+ 1 + M):
        board, pb = nextboard(board, pb, ua, r, gamma, K, show=False)
        if n >= 100 and n % di == 0 and n <= ul:
            x += hflatten(pb)
        if (n - M) >= 100 and (n - M) % di == 0 and (n - M) <= ul:
            y.append([np.sum(pb)])
    return x, y

def simdata(datapoints, di, N, M, r, gamma):
    # Simulation to extract data
    start_time = time.time()
    nsim = int(datapoints/(1000/di))
    for nm in range(nsim):
        print(str(nm) + "/" + str(nsim))
        x, y = datasimulation(50, [0, 1, 2, 3, 4, 5, 6, 7, 8], 1250, r, gamma, 0.5, 1100, M, di)
        writetocsv(x, "TXMAS.csv")
        writetocsv(y, "TYMAS.csv")
    print("--- %s seconds ---" % (time.time() - start_time))

simdata(9000, 4, 50, 10, 2.8, 0.35)