#Models.py---------------------------------------------------------------------------------------------------------

import torch.nn as nn
import torch

# simple block of convolution, batchnorm, and leakyrelu
class Block(nn.Module):
    def __init__(self, in_f, out_f):
        super(Block, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_f),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.f(x)

class DQNet(nn.Module):
    def __init__(self, num_actions):
        super(DQNet, self).__init__()

        self.convNetwork = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        self.network = nn.Sequential(
            nn.Linear(64*7*7, 256),
            nn.LeakyReLU(inplace=True),
        )

        self.valueStream = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.advantageStream = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, state):
        x = self.convNetwork(state)
        x = self.network(torch.flatten(x,1))
        value = self.valueStream(x)
        advantage = self.advantageStream(x)
        return value + (advantage - advantage.mean())

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()

        self.convNetwork = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        self.linearNet = nn.Sequential(
            nn.Linear(64*4*4, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, state):
        x = self.convNetwork(state)
        x = torch.flatten(x,1)
        x = self.linearNet(x)

        return x

#Buffers.py--------------------------------------------------------------------------------------------------------------------

import numpy as np

class PriorityReplayBuffer():
    def __init__(self, maxSize = 100000):
        self.maxSize = maxSize
        self._data = []
        self._weights = []
        self.p = 0

    def size(self):
        return len(self._data)

    def push(self, elem, weight=100):
        self._data.append(elem)
        self._weights.append(weight)
        if len(self._data) > self.maxSize:
            del self._data[0]
            del self._weights[0]

    def pushBatch(self, elem, weights):
        self._data += elem
        self._weights += weights

    def makeRoom(self):
        sizeToClear = self.size() - self.maxSize
        if self.maxSize > 10:
            self._data = self._data[sizeToClear:]
            self._weights = self._weights[sizeToClear:]

    def updateTDError(self, index, tdError):
        self._weights[index] = tdError

    def updateTDErrorBatch(self, index, tdErrors):
        for x in range(len(index)):
            self.updateTDError(int(index[x].item()), tdErrors[x].item())

    def getElem(self):
        index = np.random.choice(range(self.size()), p=np.array(self._weights)/sum(self._weights))
        return (self._data[index], index, self._weights)

    def getMiniBatch(self, size, alpha=0.5):
        probs = np.array(self._weights) ** alpha
        probs /= probs.sum()

        index = np.random.choice(range(self.size()), size, p=probs)

        return ([self._data[x] for x in index], index, [probs[x] for x in index])

#Utils.py-----------------------------------------------------------------------------------------------------------------

import torchvision,torch

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Grayscale(1),
    torchvision.transforms.Resize((84,84)),
    torchvision.transforms.ToTensor(),
])

def preprocess(image):
    return transforms(image)

#dqn3.py-----------------------------------------------------------------------------------------------------------------


import gym
import glob
import os
import numpy as np
import random
import visdom
import torch
import time
from datetime import datetime
from torchsummary import summary
import sys
import torch.multiprocessing as mp

#general settings
ENV_NAME = "Gravitar-v0"
NUM_ACTIONS = 18
EPSILON_PRIORITY_REPLAY = 1e-5
GAMMA = 0.997
MODEL_SAVE_FOLDER = "models/dqnGravitar"
VIDEO_DIR = "videos/dqnGravitar"
RESUME_TRAINING = False

#Actor params
MAX_GAME_LENGTH = 100000
DISPLAY_VIDEO_FOR_ACTOR_1 = True
N_STEPS = 3
RND_NORMALIZATION_ALPHA = 0.02
INTRINSIC_REWARD_RATIO = 0.15
PUSH_FREQUENCY=100

#learner params
TARGET_NETWORK_UPDATE_FREQUENCY = 2000
FRAMES_BEFORE_LEARNING = 128
GAMMA = 0.997
MINI_BATCH_SIZE = 32
LOGGING_BATCH_SIZE = 50 #number of games used to compute averages
BUFFER_SIZE = 200000 #had memory issues at 500k
#CHECKPOINT_FREQUENCY = 10000
USE_CUDA = True
LEARNING_RATE = 0.00015
CLEAR_BUFFER_FREQUENCY = 250
UPDATE_PREDICTOR_FREQUENCY = 4
REPLAY_BUFFER_ALPHA = 0.7
REPLAY_BUFFER_START_BETA = 0.5
REPLAY_BUFFER_END_BETA = 1
LINEAR_ANNEAL_LENGTH = 200000 #number of batches
VIDEO_EVERY = 50

LEARNING_FREQUENCY = 4#learn every 4 frames

startTime = time.time()
readableTime = datetime.utcfromtimestamp(startTime).strftime('%Y-%m-%d_%H-%M-%S')
torch.set_num_threads(1)

class Stats:
    def __init__(self):
        self.frameNumber = mp.Value("i", 0)
        self.gameNumber = mp.Value("i", 0)
        self.bestScore = mp.Value("d", 0)
        self.totalScore = mp.Value("d", 0) # total score of a batch, used to log averages

def linearAnneal(num, start, end, length):
    e = (num/length) *(end-start) + start
    return max(min(e, end), start)

def trainBatch(q, q2, buffer, device, optimizer, batchNum, stats, beta=0.4):
    #_oldsObs, _obs, _done, _action, _reward, index, probs = buffer.getMiniBatch(MINI_BATCH_SIZE)

    samples, index, probs = buffer.getMiniBatch(MINI_BATCH_SIZE, REPLAY_BUFFER_ALPHA)
    _oldObs, _obs, _reward, _done, _action = zip(*samples)

    weights = (MINI_BATCH_SIZE * torch.tensor(probs).to(device)) **(-beta)
    weights /= weights.max()
   
    action = torch.tensor(_action).to(device, dtype=torch.long)
    done = torch.tensor(_done).to(device, dtype=torch.float)
    reward = torch.tensor(_reward).to(device)
    oldsObs = torch.cat(_oldObs).to(device, dtype=torch.float)
    obs = torch.cat(_obs).to(device, dtype=torch.float)

    response = q(oldsObs)
    qsa = response.gather(1, action.unsqueeze(1)).squeeze(1)
    target = reward + GAMMA*q2(obs).max(1)[0] * (1 - done)
    loss = (qsa - target.detach()).pow(2) * weights
    
    prios = loss.detach().clone() + EPSILON_PRIORITY_REPLAY

    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(q.parameters(), 1.0)
    optimizer.step()

    buffer.updateTDErrorBatch(index, prios.cpu().detach())

    stats.loss.value += loss.item()

def trainBatchPredictor(predictor, target, buffer, device, optimizer, stats):
    samples, _, _ = buffer.getMiniBatch(MINI_BATCH_SIZE)
    _oldObs, _, _, _, _ = zip(*samples)
    oldsObs = torch.cat(_oldObs).to(device, dtype=torch.float)[:,3].unsqueeze(1)

    y = target(oldsObs)
    x = predictor(oldsObs)
    loss = (x - y.detach()).pow(2)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    stats.rndLoss.value += loss.detach().item()

def visdomReset(vis):
    vis.line(X=[0], Y=[0], win="Loss", opts=dict(title="Loss"), )
    vis.line(X=[0], Y=[0], win="RNDLoss", opts=dict(title="RNDLoss"), )

def visdomUpdate(stats, vis, beta=REPLAY_BUFFER_START_BETA):
    vis.line(X=[stats.batchNumber.value/LOGGING_BATCH_SIZE], Y=[stats.loss.value/LOGGING_BATCH_SIZE], win="Loss", opts=dict(title="Loss"), update="append")
    vis.line(X=[stats.batchNumber.value/LOGGING_BATCH_SIZE], Y=[stats.rndLoss.value/(LOGGING_BATCH_SIZE/UPDATE_PREDICTOR_FREQUENCY)], win="RNDLoss", opts=dict(title="RNDLoss"), update="append")
    stats.loss.value = 0
    stats.rndLoss.value = 0

    duration = time.time() - startTime
    vis.text("Duration: {}<br> frameNumber: {}<br> gameNumber: {}<br> batchNumber: {}<br>fps: {}<br>beta: {}".format(duration, stats.frameNumber.value, stats.gameNumber.value,stats.batchNumber.value , stats.frameNumber.value/duration, beta), win="E", opts=dict(title="Epsilon"))            

def actor(e):
    print("is cuda avaliable: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available() and USE_CUDA:  
        dev = torch.device("cuda:0")
    else:  
        dev = torch.device("cpu")

    localQ = DQNet(NUM_ACTIONS).to(dev)
    q2 = DQNet(NUM_ACTIONS).to(dev)
    q2.load_state_dict(localQ.state_dict())
    localFrameNumber = 0
    n_episode = 0

    stats = Stats()


    localTarget = Predictor().to(dev)
    localPred = Predictor().to(dev)

    optimizer = torch.optim.Adam(localQ.parameters(), lr=LEARNING_RATE)
    rndOptimizer = torch.optim.Adam(localPred.parameters(), lr=LEARNING_RATE)

    buffer = PriorityReplayBuffer(BUFFER_SIZE)
    tempBuffer = []
    qs = []
    marking = []
    batchNum = 0

    print("Starting actor Thread")   

    name1 = "Actor 1"
    name2 = name1 + " Score"
    vis = visdom.Visdom(port=12345, env="rl")
    visdomReset(vis)

    vis.line(X=[0], Y=[0], win=name2, opts=dict(title=name2), )
    env = gym.make(ENV_NAME)
    env = gym.wrappers.Monitor(env, VIDEO_DIR, video_callable=lambda episode_id: (episode_id%VIDEO_EVERY)==0,force=True)

    seed = 742
    torch.manual_seed(seed)
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env.action_space.seed(seed)

    normNumSteps = 0
    #initialize norm parameters
    targetVar = torch.zeros(256).to(dev)
    targetMu = torch.zeros(256).to(dev)
    predVar = torch.zeros(256).to(dev)
    predMu = torch.zeros(256).to(dev)
    done = False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        obs,reward,done,inf = env.step(action)
        obsP = preprocess(obs).unsqueeze(0).to(dev)
        normNumSteps += 1

        targetX = localTarget(obsP)
        predX = localPred(obsP)
        if normNumSteps == 1: #first run
            targetMu = targetX
            predMu = predX
        else:#calculate exponential rolling averages
            targetDiff = targetX - targetMu
            predDiff = predX - predMu

            targetMu = targetX * RND_NORMALIZATION_ALPHA + (1-RND_NORMALIZATION_ALPHA) * targetMu
            predMu = predX * RND_NORMALIZATION_ALPHA + (1-RND_NORMALIZATION_ALPHA) * predMu

            targetVar = (1 - RND_NORMALIZATION_ALPHA) * (targetVar  + RND_NORMALIZATION_ALPHA * targetDiff.pow(2))
            predVar = (1 - RND_NORMALIZATION_ALPHA) * (predVar  + RND_NORMALIZATION_ALPHA * predDiff.pow(2))
            
    while True:
        obs = env.reset()

        if DISPLAY_VIDEO_FOR_ACTOR_1:
            vis.image(obs.transpose(2,0,1), win=name1, opts=dict(title=name1))

        #use a random action to fill initial observation stack
        obsP = preprocess(obs).unsqueeze(0)

        initialObs = [obsP]
        act = env.action_space.sample()
        for _ in range(3):
            obs,reward,done,inf = env.step(act)
            obsP = preprocess(obs).unsqueeze(0)
            initialObs.append(obsP)
        obsStack = torch.cat(initialObs, dim=1).to(dev) #initial position

        done = False
        score = 0

        #array of trajectories,
        #qValues just used to calculate tdError estimate
        #obsStack - n*4*84*84 - stack of 4 frame images

        index = 0 #counts frame number in current game.

        while not done and index < MAX_GAME_LENGTH:    
            response = localQ(obsStack).squeeze()

            qValue = response.max().item()
            action = 0

            if random.random() < e:
                action = env.action_space.sample()
                qValue = response[action].item()
            else:
                action = response.argmax().item()

            oldObStack = obsStack.clone().cpu()
            
            obs,reward,done,inf = env.step(action)
            score+=reward #update total undiscounted reward for an episode

            if DISPLAY_VIDEO_FOR_ACTOR_1:
                vis.image(obs.transpose(2,0,1), win=name1, opts=dict(title=name1))

            obsP = preprocess(obs).unsqueeze(0).to(dev)

            #RND stuff
            targetX = localTarget(obsP)
            predX = localPred(obsP)
            targetXNorm = ((targetX - targetMu)/targetVar.sqrt()).detach()
            predXNorm = ((predX - predMu)/predVar.sqrt()).detach()
            intrinsicReward = (targetXNorm - predXNorm).pow(2).mean().item()
            reward += intrinsicReward*INTRINSIC_REWARD_RATIO
            
            targetDiff = (targetX - targetMu).detach()
            predDiff = (predX - predMu).detach()
            targetMu = (targetX * RND_NORMALIZATION_ALPHA + (1-RND_NORMALIZATION_ALPHA) * targetMu).detach()
            predMu = (predX * RND_NORMALIZATION_ALPHA + (1-RND_NORMALIZATION_ALPHA) * predMu).detach()
            targetVar = ((1 - RND_NORMALIZATION_ALPHA) * (targetVar  + RND_NORMALIZATION_ALPHA * targetDiff.pow(2))).detach()
            predVar = ((1 - RND_NORMALIZATION_ALPHA) * (predVar  + RND_NORMALIZATION_ALPHA * predDiff.pow(2))).detach()

            obsStack = torch.cat((obsStack[:,1:], obsP), dim=1)

            if index == MAX_GAME_LENGTH - 1:
                done = True #intrinsic done signal for early termination

            tempBuffer.append((oldObStack.detach(),obsStack.cpu().detach(),reward,done,action))

            qs.append(qValue)

            localFrameNumber += 1

            index += 1    

            if localFrameNumber % PUSH_FREQUENCY == 0:
                #using n step q learning to generate approximation of tderror (uses n step lookahead in learner thread) for prios
                prios = []
                sendBuffer = []
                for x in range(len(qs) - N_STEPS):
                    #reward = qs[x+N_STEPS] #calculates n step lookahead
                    rewardTraj = 0
                    endPos = x+N_STEPS-1
                    for y in range(N_STEPS):
                        pos = x + y
                        rewardTraj +=  (GAMMA**y) * tempBuffer[pos][2] 
                        if tempBuffer[pos][3]:
                            endPos = pos
                            break

                    sendBuffer.append((tempBuffer[x][0],tempBuffer[endPos][1],rewardTraj,tempBuffer[x][3],tempBuffer[x][4]))

                    if tempBuffer[endPos][3]:
                        tdError = rewardTraj - qs[x]
                    else:
                        tdError = rewardTraj- qs[x] + GAMMA * qs[endPos]

                    prios.append(abs(tdError))

                buffer.pushBatch(sendBuffer, prios)

                qs = qs[len(sendBuffer):]
                tempBuffer = tempBuffer[len(sendBuffer):]

            if buffer.size() > FRAMES_BEFORE_LEARNING and localFrameNumber % LEARNING_FREQUENCY == 0:
                batchNum += 1
                stats.batchNumber.value = batchNum
                beta = linearAnneal(batchNum, REPLAY_BUFFER_START_BETA, REPLAY_BUFFER_END_BETA, LINEAR_ANNEAL_LENGTH)
                trainBatch(localQ,q2,buffer,dev,optimizer,batchNum, stats, beta)
                if batchNum % UPDATE_PREDICTOR_FREQUENCY == 0: #update RND intrinsic reward
                    trainBatchPredictor(localPred, localTarget, buffer, dev, rndOptimizer, stats)

                if batchNum % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                    q2.load_state_dict(localQ.state_dict())
                    torch.save({"model": localQ.state_dict(),
                                "pred" : localPred.state_dict(),
                                "target" : localTarget.state_dict()},
                                MODEL_SAVE_FOLDER+"\\"+readableTime+"_"+str(batchNum+1))
        
                if batchNum % CLEAR_BUFFER_FREQUENCY  == 0:
                    buffer.makeRoom()
                
                if batchNum % LOGGING_BATCH_SIZE == 0:
                    stats.gameNumber.value = n_episode
                    stats.frameNumber.value = localFrameNumber
                    visdomUpdate(stats,vis, beta)

        n_episode += 1

        if score > stats.bestScore.value:
            stats.bestScore.value = score   

        vis.line(X=[n_episode-1], Y=[score], win=name2, opts=dict(title=name2), update="append")  

        marking.append(score)
        if n_episode%100 == 0:
            print("marking, episode: {}, score: {:.1f}, mean_score: {:.2f}, std_score: {:.2f}".format(
            n_episode, score, np.array(marking).mean(), np.array(marking).std()))
            marking = []              

class Stats:
    def __init__(self):
        self.frameNumber = mp.Value("i", 0)
        self.gameNumber = mp.Value("i", 0)
        self.batchNumber = mp.Value("i", 0)
        self.bestScore = mp.Value("d", 0)
        self.loss = mp.Value("d", 0)
        self.rndLoss = mp.Value("d", 0)

if __name__ == "__main__":
    actor(0.01)

#This is a dqn network with many extensions from rainbow (prioritized replay buffer, dueling network, 
#double q network, n step learning)
#And random network distillation as an exploration bonus. Data is logged using visdom. Default port is 12345.