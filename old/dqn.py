import gym
import numpy as np
import random
import visdom
import torch
import time
from datetime import datetime
from buffers import *
from models import *

vis = visdom.Visdom(port=12345)

USE_CUDA = False
MODEL_SAVE_FOLDER = "dqnModel"

env = gym.make("CartPole-v0")
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

#derived from https://github.com/higgsfield/RL-Adventure/blob/master/4.prioritized%20dqn.ipynb
def tdUpdate(actionNetwork, evaluationNetwork, buffer, batchSize, beta=0.4):
    global losses

    samples, index, probs = buffer.getMiniBatch(batchSize)
    oldObs, obs, action, reward, done = zip(*samples)

    weights = (len(probs) * probs[index]) **(-beta)

    print(actionNetwork(torch.tensor(oldObs, dtype=torch.float32).to(device)).size())
    print(torch.tensor(action).unsqueeze(1).size())

    actual = actionNetwork(torch.tensor(oldObs, dtype=torch.float32).to(device)).gather(1, torch.tensor(action).unsqueeze(1)).squeeze(1)
    target = torch.tensor(reward, dtype=torch.float32).to(device) + GAMMA * evaluationNetwork(torch.tensor(obs, dtype=torch.float32).to(device)).max(1)[0] * (1 -torch.tensor(done, dtype=torch.float32).to(device))

    loss = (actual - target.detach()).pow(2) * torch.tensor(weights, dtype=torch.float32).to(device)
    
    prios = loss + 1e-5 #ensures prios >0 when actual = target

    loss = loss.mean()
    losses.append(loss.item())
    optimzer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(actionNetwork.parameters(), 1.0)
    optimzer.step()

    buffer.updateTDErrorBatch(index, prios.cpu().detach().numpy())


Q = DQNet()  #main
Q2 = DQNet() #evaluation
Q2.load_state_dict(Q.state_dict())

Q.to(device)
Q2.to(device)

buffer = PriorityReplayBuffer()

criterion = torch.nn.MSELoss(reduction='sum')
optimzer = torch.optim.RMSprop(Q.parameters(), lr=0.0001)

alpha = 0.1

NUM_FRAMES = 10000000
frameNumber = 0
gameNumber = 0
STEPS_BEFORE_LEARNING = 50000

BATCH_SIZE = 50
MINI_BATCH_SIZE = 32
batch_score = 0
numStates = 0

#linear epsilon
endE = 0.1
startE = 1
linearEpsilonLength = 100000

TARGET_NETWORK_UPDATE_FREQUENCY = 10000 #copy weights every 50 episodes
averageScore = []
greedyScore = []
epsilonGraph = []
lossGraph = []

startTime = time.time()
readableTime = datetime.utcfromtimestamp(startTime).strftime('%Y-%m-%d_%H-%M-%S')

def linearEpsilon(frameNumber, start, end, length):
    e = (frameNumber/length) *(end-start) + start
    return min(max(e, end), start)

def epsilon_greedy(q,obs, epsilon): #returns action,
    if random.random() > epsilon:
        return max_action_value(q,obs)
    return env.action_space.sample()

def max_action_value(q, state): # action
    answer = q(torch.tensor(state, dtype=torch.float32).to(device))
    arr = answer.cpu().detach()
    return np.argmax(arr).item()

while frameNumber < NUM_FRAMES:
    gameNumber += 1

    e = linearEpsilon(frameNumber - STEPS_BEFORE_LEARNING, startE, endE, linearEpsilonLength)

    g = 0 # total reward
    obs =  env.reset()
    done = False
    while not done:
        frameNumber += 1

        action = epsilon_greedy(Q,obs,e)
        oldObs = obs

        obs,reward,done,inf = env.step(action)

        buffer.push((oldObs, obs, action, reward, done))
        g += reward
        numStates += 1

        if buffer.size() > STEPS_BEFORE_LEARNING:
            tdUpdate(Q, Q2, buffer, MINI_BATCH_SIZE)


        if frameNumber % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
            Q2.load_state_dict(Q.state_dict())


    batch_score += g/BATCH_SIZE
    if gameNumber % BATCH_SIZE == 0 and gameNumber != 0:
        averageScore.append(batch_score)
        vis.line(averageScore, [x for x in range(len(averageScore))], win="Score", opts=dict(title="Average Score"))
        batch_score = 0

        if(len(losses)):
            lossGraph.append(sum(losses)/len(losses))
            losses = []
            vis.line(lossGraph, [x for x in range(len(lossGraph))], win="Loss", opts=dict(title="Average Loss"))
            torch.save(Q, MODEL_SAVE_FOLDER+"\\"+readableTime+"_"+str(len(averageScore)))

        duration = time.time() - startTime
        vis.text("Duration: {}<br>Avg Time:{}<br> gameNumber: {}<br> frameNumber: {}<br> epsilon: {}".format(duration, duration/(numStates), gameNumber, frameNumber, e), win="E", opts=dict(title="Epsilon"))

print(Q)
