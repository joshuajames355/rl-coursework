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
STEPS_BEFORE_LEARNING = 128
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

DISPLAY_VIDEO = True

TRAINING_STEPS_FREQUENCY = 4

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

print("is cuda avaliable: {}".format(torch.cuda.is_available()))
if torch.cuda.is_available() and USE_CUDA:  
    device = "cuda:0"
else:  
    device = "cpu"  

def trainBatch(q, q2, buffer, optimizer):
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


def actor(e=0.01):
    localQ = R2D2Net(4).to(device)
    localFrameNumber = 0
    batchNumber = 0

    buffer = PriorityReplayBuffer()
    optimizer = torch.optim.RMSprop(localQ.parameters(), lr=0.0001)
    q2 = R2D2Net(4).to(device) #evaluation
    q2.load_state_dict(localQ.state_dict())

    gameNumber = 0
    env = gym.make(ENV_NAME)

    print("Starting actor Thread")
    vis.line(X=[0], Y=[0], win="Score", opts=dict(title="Score"))
    
    while True:
        obs = env.reset()
        if DISPLAY_VIDEO:
            vis.image(obs.transpose(2,0,1), win="Video", opts=dict(title="video"))

        obs = preprocess(obs).unsqueeze(0).to(device)
        done = False
        score = 0

        history = localQ.getNewHistory()
        history = (history[0].to(device),history[1].to(device))
        #array of trajectories,
        #format of trajectors is (initialHistory, initialObs, finalObs, action, reward, done, qValues)
        #qValues just used to calculate tdError estimate
        buffers = [((history[0].clone().detach(), history[1].clone().detach() ), [], [], [], [], [], [])]
        index = 0
        while not done and index < MAX_GAME_LENGTH:        
            action, history = localQ(obs, history)
            action = action.squeeze()

            qValue = action.max().item()
            action = action.argmax().cpu().item()

            if random.random() < e:
                action = env.action_space.sample()

            oldObs = obs
            
            obs,reward,done,inf = env.step(action)
            if DISPLAY_VIDEO:
                vis.image(obs.transpose(2,0,1), win="Video", opts=dict(title="video"))

            obs = preprocess(obs).unsqueeze(0).to(device)

            score+=reward #update total undiscounted reward for an episode

            localFrameNumber += 1

            index += 1
            for x in range(len(buffers)):
                buffers[x][1].append(oldObs.cpu().squeeze(0).numpy())
                buffers[x][2].append(obs.cpu().squeeze(0).numpy())
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

                    buffer.push(newData, prios)
                    break #cant be more than one to remove per frame
                    #push to queue

            if index % EPISODE_OVERLAP == 0:
                buffers += [((history[0].clone().detach(), history[1].clone().detach() ), [], [], [], [], [],[])]

            if buffer.size() > STEPS_BEFORE_LEARNING and localFrameNumber % TRAINING_STEPS_FREQUENCY == 0:
                trainBatch(localQ, q2, buffer, optimizer)
                if batchNumber % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                    print("UPDATE Q2!")
                    q2.load_state_dict(localQ.state_dict())
                batchNumber += 1

        gameNumber += 1

        vis.line(X=[gameNumber], Y=[score], win="Score", opts=dict(title="Score"), update="append")

        duration = time.time() - startTime
        vis.text("Duration: {}<br> frameNumber: {}<br> gameNumber: {}<br> batchNumber: {}<br>fps: {}".format(duration, localFrameNumber, gameNumber ,batchNumber , localFrameNumber/duration), win="E", opts=dict(title="Epsilon"))            
            

if __name__ == "__main__":
    actor()

    #test()