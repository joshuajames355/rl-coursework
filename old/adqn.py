import gym
import numpy as np
import random
import visdom
import torch
import time
from datetime import datetime
from buffers import *
from models import *
import torch.multiprocessing as mp

vis = visdom.Visdom(port=12345)

USE_CUDA = False
MODEL_SAVE_FOLDER = "aqn"
envName = "CartPole-v0"

#env = gym.wrappers.Monitor(env, 'videos', force = True)

#env.render()

print("is cuda avaliable: {}".format(torch.cuda.is_available()))
if torch.cuda.is_available() and USE_CUDA:  
    dev = "cuda:0"
else:  
    dev = "cpu"  
device = torch.device(dev)

GAMMA = 0.99

losses = []


Q = QNet()  #main
Q2 = QNet() #evaluation
Q2.load_state_dict(Q.state_dict())

Q.to(device)
Q2.to(device)


alpha = 0.1

NUM_FRAMES = 10000000
#frameNumber = 0
#gameNumber = 0

BATCH_SIZE = 5000
batch_score = 0
numStates = 0

NUM_THREADS = 12

TMAX = 50

TARGET_NETWORK_UPDATE_FREQUENCY = 10000 #copy weights every 50 episodes

startTime = time.time()
readableTime = datetime.utcfromtimestamp(startTime).strftime('%Y-%m-%d_%H-%M-%S')

def linearEpsilon(frameNumber, start, end, length):
    e = (frameNumber/length) *(end-start) + start
    return min(max(e, end), start)

def epsilon_greedy(q,obs, epsilon, env): #returns action,
    if random.random() > epsilon:
        return max_action_value(q,obs, env)
    return env.action_space.sample()

def max_action_value(q, state, env): # action
    answer = q(torch.tensor(state, dtype=torch.float32).to(device))
    arr = answer.cpu().detach()
    return np.argmax(arr).item()

def train(q,q2, e, stats):
    env = gym.make("CartPole-v0")
    optimzer = torch.optim.RMSprop(Q.parameters(), lr=0.0001)

    while True:
        obs = env.reset()
    
        done = False
        g = 0
        score = 0
        while not done:
            states = []
            t = 0
            
            while (not done) and t < TMAX:
                t += 1

                action = epsilon_greedy(q,obs,e, env)

                oldObs = obs
                obs,reward,done,inf = env.step(action)
                states.append((oldObs,reward,done,inf, action))

                g+=reward #update total score
                score += reward

                with stats.frameNumber.get_lock():
                    if stats.frameNumber.value % TARGET_NETWORK_UPDATE_FREQUENCY == 0 and stats.frameNumber.value > 0:
                        q2.load_state_dict(q.state_dict())

                    stats.frameNumber.value += 1

            optimzer.zero_grad()

            R = 0
            if not done:
                R = max_action_value(q2, obs, env)
            
            for obs,reward,done,inf, action in states[::-1]:
                R = reward + GAMMA * R

                value = q(torch.tensor(obs, dtype=torch.float32).to(device))[action]


                loss = (R - value).pow(2)
                loss.backward()

            optimzer.step()
            optimzer.zero_grad()
            
        with stats.gameNumber.get_lock():
            stats.gameNumber.value += 1

            if(stats.gameNumber.value % BATCH_SIZE == 0):
                visdomUpdate(stats)

        with stats.totalScore.get_lock():
            stats.totalScore.value += score

        if score > stats.bestScore.value:
            stats.bestScore.value = score
            
def visdomUpdate(stats):
    vis.line(X=[stats.gameNumber.value/BATCH_SIZE], Y=[stats.totalScore.value/BATCH_SIZE], win="Score", opts=dict(title="Average Score"), update="append")
    vis.line(X=[stats.gameNumber.value/BATCH_SIZE], Y=[stats.bestScore.value], win="BScore", opts=dict(title="Best Score"), update="append")
    stats.totalScore.value = 0

    #torch.save(Q, MODEL_SAVE_FOLDER+"\\"+readableTime+"_"+str(len(scores)))

    duration = time.time() - startTime
    vis.text("Duration: {}<br> frameNumber: {}<br> gameNumber: {}".format(duration, stats.frameNumber.value, stats.gameNumber.value), win="E", opts=dict(title="Epsilon"))

class Stats:
    def __init__(self):
        self.frameNumber = mp.Value("i", 0)
        self.gameNumber = mp.Value("i", 0)
        self.bestScore = mp.Value("d", 0)
        self.totalScore = mp.Value("d", 0) # total score of a batch, used to log averages

if __name__ == "__main__":
    stats = Stats()

    Q.share_memory()
    Q2.share_memory()
    processes = []
    for x in range(NUM_THREADS-1):
        p = mp.Process(target=train, args=(Q, Q2, (0.5*x)/NUM_THREADS, stats),) #epsilon varies from 0 to 0.5
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
