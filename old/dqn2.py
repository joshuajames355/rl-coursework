import gym
import numpy as np
import torch.nn as nn
import random
import visdom
import torch
import time
from datetime import datetime
from torchsummary import summary
import torch.multiprocessing as mp
import sys
from utils import preprocess

USE_CUDA = True
MODEL_SAVE_FOLDER = "adqnModel"

GAMMA = 0.997

NUM_THREADS = 5 #must be atleast 2

ENV_NAME = "Breakout-v0"

NUM_ACTIONS = 4

#Actor params
MAX_GAME_LENGTH = 10000
COPY_NETWORK_FREQUENCY = 500 #each actor thread copies the network every n frames(counted locally)
PUSH_FREQUENCY = 100
DISPLAY_VIDEO = True

#learner params
TARGET_NETWORK_UPDATE_FREQUENCY = 25 #copy weights every 50 batches, for learner thread
N_STEPS = 6 #number of steps to use when calculating target
FRAMES_BEFORE_LEARNING = 2500
GAMMA = 0.997
MINI_BATCH_SIZE = 64
LOGGING_BATCH_SIZE = 15 #number of games used to compute averages
BUFFER_SIZE = 50000
CHECKPOINT_FREQUENCY = 50

startTime = time.time()
readableTime = datetime.utcfromtimestamp(startTime).strftime('%Y-%m-%d_%H-%M-%S')

class PriorityReplayBuffer():
    def __init__(self, maxSize = 100000):
        self.maxSize = maxSize

        self._oldObs = torch.empty(1,84,84,4) #n*84*84*4
        self._obs = torch.empty(1,84,84,4) #n*84*84*4
        self._done = torch.empty(1) #n
        self._reward = torch.empty(1) #n
        self._action = torch.empty(1) #n

        self._weights = []

    def size(self):
        return len(self._weights)

    def push(self, oldObs, obs, done, reward, action, weights):
        if len(self._weights) == 0:
            self._oldObs = oldObs.clone()
            self._obs = obs.clone()
            self._done = done.clone()
            self._reward = reward.clone()
            self._action = action.clone()
            self._weights = weights   
        else:
            self._oldObs = torch.cat((self._oldObs, oldObs))
            self._obs = torch.cat((self._obs, obs))
            self._done = torch.cat((self._done, done))
            self._reward = torch.cat((self._reward, reward))
            self._action = torch.cat((self._action, action))
            self._weights += weights

    def makeRoom(self):
        sizeToClear = len(self._weights) - self.maxSize

        self._oldObs = self._oldObs[sizeToClear:]
        self._obs = self._obs[sizeToClear:]
        self._done = self._done[sizeToClear:]
        self._reward = self._reward[sizeToClear:]
        self._action = self._action[sizeToClear:]
        self._weights = self._weights[sizeToClear:]
        print("made room!")

    def updateTDErrorBatch(self, index, tdErrors):
        for x in range(len(index)):
            self._weights[int(index[x].item())] = tdErrors[x]

    def getMiniBatch(self, size):
        #print(self._weights)
        probs = np.array(self._weights) ** 0.6
        probs /= probs.sum()

        index = torch.tensor(np.random.choice(range(self.size()), size, p=probs), dtype=torch.long)

        return (self._oldObs.index_select(0, index), 
            self._obs.index_select(0, index), 
            self._done.index_select(0, index), 
            self._action.index_select(0, index), 
            self._reward.index_select(0, index), 
            index, probs)

class Stats:
    def __init__(self):
        self.frameNumber = mp.Value("i", 0)
        self.gameNumber = mp.Value("i", 0)
        self.bestScore = mp.Value("d", 0)
        self.totalScore = mp.Value("d", 0) # total score of a batch, used to log averages

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

def trainBatch(q, q2, buffer, device, optimizer, batchNum, stats):
    _oldsObs, _obs, _done, _action, _reward, index, probs = buffer.getMiniBatch(MINI_BATCH_SIZE)
    
    action = _action.to(device, dtype=torch.long).detach()
    done = _done.to(device, dtype=torch.float).detach()
    reward = _reward.to(device).detach()
    oldsObs = _oldsObs.to(device).detach()
    obs = _obs.to(device).detach()
        
    response = q(oldsObs)
    qsa = response.gather(1, action.unsqueeze(1)).squeeze(1)
    target = reward + GAMMA*q2(obs).max() * (1 - done)
    loss = (qsa - target.detach()).pow(2)# * torch.tensor(weights, dtype=torch.float32).to(device)
    
    prios = loss.detach().clone()

    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(q.parameters(), 1.0)
    optimizer.step()

    buffer.updateTDErrorBatch(index, prios.cpu().detach())

    stats.loss.value += loss.item()


def visdomReset(vis):
    vis.line(X=[0], Y=[0], win="BScore", opts=dict(title="Best Score"), )
    vis.line(X=[0], Y=[0], win="Loss", opts=dict(title="Loss"), )

def visdomUpdate(stats, vis):
    vis.line(X=[stats.gameNumber.value/LOGGING_BATCH_SIZE], Y=[stats.bestScore.value], win="BScore", opts=dict(title="Best Score"), update="append")
    vis.line(X=[stats.batchNumber.value], Y=[stats.loss.value], win="Loss", opts=dict(title="Loss"), update="append")
    stats.loss.value = 0

    duration = time.time() - startTime
    vis.text("Duration: {}<br> frameNumber: {}<br> gameNumber: {}<br> batchNumber: {}<br>fps: {}".format(duration, stats.frameNumber.value, stats.gameNumber.value,stats.batchNumber.value , stats.frameNumber.value/duration), win="E", opts=dict(title="Epsilon"))            

def learner(q, pipes, stats):
    vis = visdom.Visdom(port=12345)
    visdomReset(vis)

    buffer = PriorityReplayBuffer(BUFFER_SIZE)
    #data is (initial history, [( oldObs, obs, action, reward, done), ...])

    #wait for epsides before starting learning

    print("is cuda avaliable: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available() and USE_CUDA:  
        dev = "cuda:0"
    else:  
        dev = "cpu"  

    qCuda = q.to(dev)

    q2 = DQNet(NUM_ACTIONS).to(dev) #evaluation
    q2.load_state_dict(q.state_dict())

    optimizer = torch.optim.RMSprop(qCuda.parameters(), lr=0.001)

    print("Starting learner Thread")
    while buffer.size() < min(FRAMES_BEFORE_LEARNING, buffer.maxSize):
        for pipe in pipes:
            if pipe.poll():
                a = pipe.recv()
                b = pipe.recv()
                c = pipe.recv()
                d = pipe.recv()
                e = pipe.recv()
                f = pipe.recv()
                buffer.push(a,b,c,d,e,f)
        visdomUpdate(stats, vis)

    print("Buffer Filled, starting learning!")

    batchNum = 0

    visdomUpdate(stats, vis)

    while True:
        for pipe in pipes:
            if pipe.poll():
                a = pipe.recv()
                b = pipe.recv()
                c = pipe.recv()
                d = pipe.recv()
                e = pipe.recv()
                f = pipe.recv()
                buffer.push(a,b,c,d,e,f)

        trainBatch(qCuda, q2, buffer, dev, optimizer, batchNum, stats)

        q.load_state_dict(qCuda.state_dict())

        batchNum += 1

        stats.batchNumber.value = batchNum

        visdomUpdate(stats,vis)

        if batchNum % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
            print("UPDATE Q2!")
            q2.load_state_dict(q.state_dict())
            torch.save(q, MODEL_SAVE_FOLDER+"\\"+readableTime+"_"+str(batchNum+1))

            if buffer.size() >= BUFFER_SIZE * 1.2:
                buffer.makeRoom()
        
def actor(q , stats, queue, number, e):
    localQ = DQNet(NUM_ACTIONS)
    localQ.load_state_dict(q.state_dict())
    localFrameNumber = 0
    localGameNumber = 0

    obsBuffer = torch.empty(PUSH_FREQUENCY,4,84,84)
    oldObsBuffer = torch.empty(PUSH_FREQUENCY,4,84,84)
    rewardBuffer = torch.empty(PUSH_FREQUENCY)
    actionBuffer = torch.empty(PUSH_FREQUENCY)
    doneBuffer = torch.empty(PUSH_FREQUENCY)

    qs = []

    print("Starting actor Thread")   
    

    name1 = "Actor " + str(number)
    name2 = name1 + " Score"
    vis = visdom.Visdom(port=12345)

    vis.line(X=[0], Y=[0], win=name2, opts=dict(title=name2), )
    
    while True:
        env = gym.make(ENV_NAME)
        obs = env.reset()

        if DISPLAY_VIDEO:
            vis.image(obs.transpose(2,0,1), win=name1, opts=dict(title=name1))

        obsP = preprocess(obs).unsqueeze(0)

        done = False
        score = 0

        #array of trajectories,
        #format of trajectors is ( initialObs - 84x84x4 stack of 4 frames, finalObs - 84x84x4, action, reward, done, qValues)
        #qValues just used to calculate tdError estimate

        index = 0

        obsStack = torch.cat((obsP,obsP,obsP,obsP), dim=1) #initial position

        while not done and index < MAX_GAME_LENGTH:    
            action = localQ(obsStack)
            action = action.squeeze()

            qValue = action.max().item()
            action = action.argmax().item()

            if random.random() < e:
                action = env.action_space.sample()

            oldObStack = obsStack.clone()
            
            obs,reward,done,inf = env.step(action)

            if DISPLAY_VIDEO:
                vis.image(obs.transpose(2,0,1), win=name1, opts=dict(title=name1))

            obsP = preprocess(obs).unsqueeze(0)

            obsStack = torch.cat((obsStack, obsP), dim=1)

            obsStack = obsStack[:,1:]

            if index == MAX_GAME_LENGTH - 1:
                done = True #intrinsic done signal for early termination
        
            score+=reward #update total undiscounted reward for an episode

            oldObsBuffer[index % PUSH_FREQUENCY] = oldObStack
            obsBuffer[index % PUSH_FREQUENCY] = obsStack
            rewardBuffer[index % PUSH_FREQUENCY] = reward
            doneBuffer[index % PUSH_FREQUENCY] = done
            actionBuffer[index % PUSH_FREQUENCY] = action

            qs.append(qValue)

            localFrameNumber += 1
            with stats.frameNumber.get_lock():
                stats.frameNumber.value += 1

            if localFrameNumber % COPY_NETWORK_FREQUENCY == 0:
                localQ.load_state_dict(q.state_dict())

            index += 1    

            if localFrameNumber % PUSH_FREQUENCY == 0:
                #push buffered experiences after game finishes
                #using 1 step q learning to generate approximation of tderror (uses n step lookahead in learner thread) for prios
                prios = []
                for x in range(len(qs)):
                    tdError = 0
                    if doneBuffer[x] or x == len(qs)-1:
                        tdError = rewardBuffer[x] - qs[x]
                    else:
                        tdError = rewardBuffer[x] - qs[x] + GAMMA * qs[x+1]

                    prios.append(abs(tdError.item()))

                queue.send(oldObsBuffer)
                queue.send(obsBuffer)
                queue.send(doneBuffer)
                queue.send(rewardBuffer)
                queue.send(actionBuffer)
                queue.send(prios)
                qs = []

        localGameNumber += 1

        with stats.gameNumber.get_lock():
            stats.gameNumber.value += 1

        if score > stats.bestScore.value:
            stats.bestScore.value = score   

        vis.line(X=[localGameNumber], Y=[score], win=name2, opts=dict(title=name2), update="append")

def singleThreadedLearner():

                 

class Stats:
    def __init__(self):
        self.frameNumber = mp.Value("i", 0)
        self.gameNumber = mp.Value("i", 0)
        self.batchNumber = mp.Value("i", 0)
        self.bestScore = mp.Value("d", 0)
        self.loss = mp.Value("d", 0)

def main():
    print("main")
    
    Q = DQNet(NUM_ACTIONS)#.to(device)  #main
    Q.share_memory()

    threads = max(2,NUM_THREADS) # needs atleast 2 threads
    stats = Stats()

    processes = []

    #a variety of epsilsons values picked to be in range 0.01->0.5, skewed to the left
    epsilons = [0.5, 0.01, 0.1, 0.15, 0.12, 0.05, 0.3, 0.07, 0.4, 0.2, 0.14, 0.02, 0.25] 
    pipes = []
    for x in range(threads-1):
        parent, child = mp.Pipe(duplex=False)
        pipes.append(parent)

        p = mp.Process(target=actor, args=(Q, stats, child,x+1,epsilons[x]),) #epsilon varies from 0 to 0.5
        processes.append(p)
        p.start()

    p = mp.Process(target=learner, args=(Q, pipes, stats),)
    p.start()
    processes.append(p)    

    for each in processes:
        each.join()

def test():
    Q = DQNet(NUM_ACTIONS)#.to(device)  #main
    Q.share_memory()
    stats = Stats()
    parent, child = mp.Pipe(duplex=False)
    actor(Q, stats, child)

if __name__ == "__main__":
    main()

   # test()

