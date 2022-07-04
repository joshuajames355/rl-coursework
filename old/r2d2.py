import gym
import numpy as np
import random
import visdom
import torch
import time
from datetime import datetime
from buffers import *
from models import *
from utils import *
from torchsummary import summary
import torch.multiprocessing as mp
import sys

vis = visdom.Visdom(port=12345)

USE_CUDA = True
MODEL_SAVE_FOLDER = "r2d2Model"

GAMMA = 0.997

alpha = 0.1

NUM_THREADS = 6 #must be atleast 2

NUM_FRAMES = 10000000
frameNumber = 0
gameNumber = 0
STEPS_BEFORE_LEARNING = 250
MAX_GAME_LENGTH = 100
NUM_ACTION_REPEATS = 4
ENV_NAME = "Breakout-v0"
N_STEPS = 4 #number of steps to use when calculating target

LOGGING_BATCH_SIZE = 15 #number of games used to compute averages
MINI_BATCH_SIZE = 64

BURN_IN_LENGTH = 15 #number of actions before training to reset history
EPISODE_LENGTH = 80
EPISODE_OVERLAP = 40
REPLAY_PRIORITIZATION_MAX_FACTOR = 0.9

batch_score = 0
numStates = 0

#linear epsilon
endE = 0.1
startE = 1
linearEpsilonLength = 100000

TARGET_NETWORK_UPDATE_FREQUENCY = 20 #copy weights every 50 batches, for learner thread
COPY_NETWORK_FREQUENCY = 500#each actor thread copies the network every n frames(counted locally)
averageScore = []
greedyScore = []
epsilonGraph = []
lossGraph = []

startTime = time.time()
readableTime = datetime.utcfromtimestamp(startTime).strftime('%Y-%m-%d_%H-%M-%S')

class Stats:
    def __init__(self):
        self.frameNumber = mp.Value("i", 0)
        self.gameNumber = mp.Value("i", 0)
        self.bestScore = mp.Value("d", 0)
        self.totalScore = mp.Value("d", 0) # total score of a batch, used to log averages

def trainBatch(q, q2, buffer, device, optimizer, batchNum):
    samples, index, probs = buffer.getMiniBatch(MINI_BATCH_SIZE)
    #samples is an [BatchSize]  containing
    #(history, [obs...], [oldObs...], [])
    history, _oldsObs, _obs, _action, _reward, _done, = zip(*samples)
    #_oldObs is now [batchSize, epsideNum, ...]

    _action = torch.tensor(_action, dtype=torch.long).to(device)
    _done = torch.tensor(_done, dtype=torch.float32).to(device)
    _reward = torch.tensor(_reward).to(device)
    _oldsObs = torch.tensor(_oldsObs).to(device)
    _obs = torch.tensor(_obs).to(device)

    h_0, c_0 = zip(*history)
    h_0 = torch.cat( h_0,1).to(device)
    c_0 = torch.cat( c_0,1).to(device)

    history = (h_0, c_0)    

    maxTDError = torch.zeros(MINI_BATCH_SIZE).to(device) # used to calculate new prios
    totalTDError = torch.zeros(MINI_BATCH_SIZE).to(device) 

    averageLoss = 0
    for x in range(EPISODE_LENGTH):
        oneTensor = torch.tensor([x]).to(device)
        action = _action.index_select(1,oneTensor).squeeze(1)
        reward = _reward.index_select(1,oneTensor).squeeze(1)
        oldsObs = _oldsObs.index_select(1,oneTensor).squeeze(1)
        
        response, history = q(oldsObs, history, train=True)
              
        if x > BURN_IN_LENGTH:
            qsa = response.gather(1, action.unsqueeze(1)).squeeze(1)

            #start n step lookahead for targeet

            lookAheadStart = min(EPISODE_LENGTH, x + N_STEPS)

            obs = _obs.index_select(1,oneTensor).squeeze(1)
            done = _done.index_select(1,oneTensor).squeeze(1)
            reward = _reward.index_select(1,oneTensor).squeeze(1)
            target = reward + GAMMA*q2(obs, history)[0].max(1)[0] * (1 - done)

            for y in range(lookAheadStart-1, x-1, -1):
                oneTensor = torch.tensor([y]).to(device)
                reward = _reward.index_select(1,oneTensor).squeeze(1)

                target = target * GAMMA + reward

            #finish n step lookahead


            loss = (qsa - target.detach()).pow(2)# * torch.tensor(weights, dtype=torch.float32).to(device)
    
            maxTDError = torch.max(maxTDError, loss)
            totalTDError += loss

            loss = loss.mean()
            averageLoss+=loss.cpu().item()
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_value_(q.parameters(), 1.0)
            optimizer.step()

    prios = REPLAY_PRIORITIZATION_MAX_FACTOR * maxTDError + (1-REPLAY_PRIORITIZATION_MAX_FACTOR) * totalTDError/(EPISODE_LENGTH-BURN_IN_LENGTH)
    buffer.updateTDErrorBatch(index, prios.cpu().detach().numpy())

    vis.line(X=[batchNum+1], Y=[averageLoss/EPISODE_LENGTH], win="Loss", opts=dict(title="Loss"), update="append")
    torch.save(q, MODEL_SAVE_FOLDER+"\\"+readableTime+"_"+str(batchNum+1))

def visdomReset():
    vis.line(X=[0], Y=[0], win="Score", opts=dict(title="Average Score"), )
    vis.line(X=[0], Y=[0], win="BScore", opts=dict(title="Best Score"), )
    vis.line(X=[0], Y=[0], win="Loss", opts=dict(title="Loss"), )

def visdomUpdate(stats):
    vis.line(X=[stats.gameNumber.value/LOGGING_BATCH_SIZE], Y=[stats.totalScore.value/LOGGING_BATCH_SIZE], win="Score", opts=dict(title="Average Score"), update="append")
    vis.line(X=[stats.gameNumber.value/LOGGING_BATCH_SIZE], Y=[stats.bestScore.value], win="BScore", opts=dict(title="Best Score"), update="append")
    stats.totalScore.value = 0

    duration = time.time() - startTime
    vis.text("Duration: {}<br> frameNumber: {}<br> gameNumber: {}<br> batchNumber: {}<br>fps: {}".format(duration, stats.frameNumber.value, stats.gameNumber.value,stats.batchNumber.value , stats.frameNumber.value/duration), win="E", opts=dict(title="Epsilon"))            

def learner(q, inQueue, stats):
    buffer = PriorityReplayBuffer()
    #data is (initial history, [( oldObs, obs, action, reward, done), ...])

    #wait for epsides before starting learning

    print("is cuda avaliable: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available() and USE_CUDA:  
        dev = "cuda:0"
    else:  
        dev = "cpu"  

    q.to(dev)

    q2 = R2D2Net(4) #evaluation
    q2.load_state_dict(q.state_dict())
    q2.to(dev)

    optimizer = torch.optim.RMSprop(q.parameters(), lr=0.0001)

    print("Starting learner Thread")
    while buffer.size() < min(STEPS_BEFORE_LEARNING, buffer.maxSize):
            data, prio = inQueue.get()
            buffer.push(data, prio)

    print("Buffer Filled, starting learning!")

    batchNum = 0

    while True:
        while not inQueue.empty(): #fill buffer with incoming episodes
            data, prio = inQueue.get()
            buffer.push(data, prio)

        trainBatch(q, q2, buffer, dev, optimizer, batchNum)

        batchNum += 1

        stats.batchNumber.value = batchNum

        if batchNum % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
            print("UPDATE Q2!")
            q2.load_state_dict(q.state_dict())

def actor(q , stats, queue, e=0.01):
    localQ = R2D2Net(4)
    localQ.load_state_dict(q.state_dict())
    localFrameNumber = 0

    print("Starting actor Thread")
    
    while True:
        env = gym.make(ENV_NAME)
        obs = env.reset()
        obs = preprocess(obs).unsqueeze(0)
        done = False
        score = 0

        history = localQ.getNewHistory()
        #array of trajectories,
        #format of trajectors is (initialHistory, initialObs, finalObs, action, reward, done, qValues)
        #qValues just used to calculate tdError estimate
        buffers = [((history[0].clone().detach(), history[1].clone().detach() ), [], [], [], [], [], [])]
        index = 0
        while not done and index < MAX_GAME_LENGTH:        
            action, history = localQ(obs, history)
            action = action.squeeze()

            qValue = action.max().item()
            action = action.argmax().item()

            if random.random() < e:
                action = env.action_space.sample()

            oldObs = obs
            for _ in range(NUM_ACTION_REPEATS):
                obs,reward,done,inf = env.step(action)
                if done:
                    break
            obs = preprocess(obs).unsqueeze(0)

            score+=reward #update total undiscounted reward for an episode

            localFrameNumber += 1
            with stats.frameNumber.get_lock():
                stats.frameNumber.value += 1

            if localFrameNumber % COPY_NETWORK_FREQUENCY:
                localQ.load_state_dict(q.state_dict())

            index += 1
            for x in range(len(buffers)):
                buffers[x][1].append(oldObs.squeeze(0).numpy())
                buffers[x][2].append(obs.squeeze(0).numpy())
                buffers[x][3].append(action)
                buffers[x][4].append(reward)
                buffers[x][5].append(done)
                buffers[x][6].append(qValue)

            for x in range(len(buffers)):
                if len(buffers[x][1]) >= EPISODE_LENGTH:
                    maxQValue = 0
                    meanQValue = 0
                    
                    for y in range(EPISODE_LENGTH-1):#just using one step as an initial approximation, no seperate target network.
                        tdError = (buffers[x][4][y] + GAMMA * buffers[x][6][y+1] - buffers[x][6][y])**2
                        #(r + gamma* q(s+1, a) - q(s,a) )**2

                        maxQValue = max(qValue, tdError)
                        meanQValue += tdError

                    prios = REPLAY_PRIORITIZATION_MAX_FACTOR * maxQValue + (1-REPLAY_PRIORITIZATION_MAX_FACTOR) * meanQValue/(EPISODE_LENGTH-1)

                    #remove qValues from buffer
                    temp = buffers.pop(x)
                    newData = (temp[0], temp[1],temp[2], temp[3],temp[4],temp[5])

                    queue.put((newData, prios))
                    break #cant be more than one to remove per frame
                    #push to queue

            if index % EPISODE_OVERLAP == 0:
                buffers += [((history[0].clone().detach(), history[1].clone().detach() ), [], [], [], [], [],[])]

        with stats.gameNumber.get_lock():
            stats.gameNumber.value += 1
            if stats.gameNumber.value % LOGGING_BATCH_SIZE == 0:
                visdomUpdate(stats)

        with stats.totalScore.get_lock():
            stats.totalScore.value += score

        if score > stats.bestScore.value:
            stats.bestScore.value = score
            

class Stats:
    def __init__(self):
        self.frameNumber = mp.Value("i", 0)
        self.gameNumber = mp.Value("i", 0)
        self.batchNumber = mp.Value("i", 0)
        self.bestScore = mp.Value("d", 0)
        self.totalScore = mp.Value("d", 0) # total score of a batch, used to log averages

def main():
    visdomReset()

    Q = R2D2Net(4)#.to(device)  #main
    Q.share_memory()

    queue = mp.SimpleQueue()

    threads = max(2,NUM_THREADS) # needs atleast 2 threads
    stats = Stats()

    processes = []

    #a variety of epsilsons values picked to be in range 0.01->0.5, skewed to the left
    epsilons = [0.01, 0.5, 0.1, 0.15, 0.12, 0.05, 0.3, 0.07, 0.4, 0.2, 0.14, 0.02, 0.25] 
    for x in range(threads-1):
        p = mp.Process(target=actor, args=(Q, stats, queue, epsilons[x]),) #epsilon varies from 0 to 0.5
        processes.append(p)
        p.start()

    learner(Q, queue, stats)

    #p = mp.Process(target=learner, args=(Q, queue),)
    #p.start()
    #processes.append(p)    

def test():
    Q = R2D2Net(4)#.to(device)  #main
    stats = Stats()
    queue = mp.SimpleQueue()

    actor(Q, stats, queue, 0.01)

if __name__ == "__main__":
    main()

    #test()