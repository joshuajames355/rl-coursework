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
import torch.multiprocessing as mp
import sys
from utils import preprocess
from buffers import PriorityReplayBuffer
from adqnModels import DQNet, Predictor

#general settings
ENV_NAME = "Gravitar-v0"
NUM_ACTIONS = 18
EPSILON_PRIORITY_REPLAY = 1e-5
GAMMA = 0.997
NUM_THREADS = 6 #must be atleast 2
MODEL_SAVE_FOLDER = "models/adqnPong"
RESUME_TRAINING = False

#Actor params
MAX_GAME_LENGTH = 100000
COPY_NETWORK_FREQUENCY = 500 #each actor thread copies the network every n frames(counted locally)
PUSH_FREQUENCY = 500
DISPLAY_VIDEO_FOR_ACTOR_1 = True
N_STEPS = 3
RND_NORMALIZATION_ALPHA = 0.02
INTRINSIC_REWARD_RATIO = 0.002

#learner params
TARGET_NETWORK_UPDATE_FREQUENCY = 2000
FRAMES_BEFORE_LEARNING = 64#10000
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
    _oldObs, _obs, _reward, _intrinsicReward, _done, _action = zip(*samples)

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

def learner(q, inQueue, stats, target, predictor):
    vis = visdom.Visdom(port=12345)
    visdomReset(vis)

    buffer = PriorityReplayBuffer(BUFFER_SIZE)

    print("is cuda avaliable: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available() and USE_CUDA:  
        dev = torch.device("cuda:0")
    else:  
        dev = torch.device("cpu")

    q2 = DQNet(NUM_ACTIONS).to(dev) #evaluation
    q2.load_state_dict(q.state_dict())

    qTraining = DQNet(NUM_ACTIONS).to(dev) #A copy of q that is kept on gpu. 
    qTraining.load_state_dict(q.state_dict())#original is kept in shared memory for other threads to access, synced every batch

    targetCuda = Predictor().to(dev)
    targetCuda.load_state_dict(target.state_dict())

    predictorCuda = Predictor().to(dev)
    predictorCuda.load_state_dict(predictor.state_dict())

    optimizer = torch.optim.Adam(qTraining.parameters(), lr=LEARNING_RATE)
    rndOptimizer = torch.optim.Adam(predictorCuda.parameters(), lr=LEARNING_RATE)

    print("Starting learner Thread")
    #wait for epsides before starting learning
    while buffer.size() < min(FRAMES_BEFORE_LEARNING, buffer.maxSize):
        buffer.pushBatch(*inQueue.get())
        visdomUpdate(stats, vis)

    print("Buffer Filled, starting learning!")

    batchNum = 0

    visdomUpdate(stats, vis)

    while True:
        try:
            while inQueue.qsize() > 5: #maybe stops the locks
                buffer.pushBatch(*inQueue.get())
        except Exception as e:
            print("Exception!")
            print(e)

        beta = linearAnneal(batchNum, REPLAY_BUFFER_START_BETA, REPLAY_BUFFER_END_BETA, LINEAR_ANNEAL_LENGTH)

        trainBatch(qTraining, q2, buffer, dev, optimizer, batchNum, stats, beta)
        q.load_state_dict(qTraining.state_dict())


        batchNum += 1
        stats.batchNumber.value = batchNum

        if batchNum % UPDATE_PREDICTOR_FREQUENCY == 0: #update RND intrinsic reward
            trainBatchPredictor(predictorCuda, targetCuda, buffer, dev, rndOptimizer, stats)
            predictor.load_state_dict(predictorCuda.state_dict())

        if batchNum % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
            q2.load_state_dict(q.state_dict())
            torch.save({"model": q.state_dict(),
                        "pred" : predictor.state_dict(),
                        "target" : target.state_dict()},
                        MODEL_SAVE_FOLDER+"\\"+readableTime+"_"+str(batchNum+1))
            print("UPDATE Q2!")
        
        if batchNum % CLEAR_BUFFER_FREQUENCY  == 0:
            buffer.makeRoom()
        
        if batchNum % LOGGING_BATCH_SIZE == 0:
            visdomUpdate(stats,vis, beta)
        
def actor(q ,target, pred, stats, queue, number, e):
    localQ = DQNet(NUM_ACTIONS)
    localQ.load_state_dict(q.state_dict())
    localFrameNumber = 0
    localGameNumber = 0

    localTarget = target#localTarget = Predictor()
    #localTarget.load_state_dict(target.state_dict())
    localPred = Predictor()
    localPred.load_state_dict(pred.state_dict())

    buffer = []

    qs = []

    print("Starting actor Thread")   

    name1 = "Actor " + str(number)
    name2 = name1 + " Score"
    vis = visdom.Visdom(port=12345)

    vis.line(X=[0], Y=[0], win=name2, opts=dict(title=name2), )
    env = gym.make(ENV_NAME)

    normNumSteps = 0
    #initialize norm parameters
    targetVar = torch.zeros(256)
    targetMu = torch.zeros(256)
    predVar = torch.zeros(256)
    predMu = torch.zeros(256)

    done = False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        obs,reward,done,inf = env.step(action)
        obsP = preprocess(obs).unsqueeze(0)
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

        if DISPLAY_VIDEO_FOR_ACTOR_1 and number == 1:
            vis.image(obs.transpose(2,0,1), win=name1, opts=dict(title=name1))

        #use a random action to fill initial observation stack
        obsP = preprocess(obs).unsqueeze(0)

        initialObs = [obsP]
        act = env.action_space.sample()
        for _ in range(3):
            obs,reward,done,inf = env.step(act)
            obsP = preprocess(obs).unsqueeze(0)
            initialObs.append(obsP)
        obsStack = torch.cat(initialObs, dim=1) #initial position

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

            oldObStack = obsStack.clone()
            
            obs,reward,done,inf = env.step(action)
            score+=reward #update total undiscounted reward for an episode

            if DISPLAY_VIDEO_FOR_ACTOR_1 and number == 1:
                vis.image(obs.transpose(2,0,1), win=name1, opts=dict(title=name1))

            obsP = preprocess(obs).unsqueeze(0)

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

            buffer.append((oldObStack,obsStack,reward,done,action))

            qs.append(qValue)

            localFrameNumber += 1
            with stats.frameNumber.get_lock():
                stats.frameNumber.value += 1

            if localFrameNumber % COPY_NETWORK_FREQUENCY == 0:
                localQ.load_state_dict(q.state_dict())
                localPred.load_state_dict(pred.state_dict())

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
                        rewardTraj +=  (GAMMA**y) * buffer[pos][2] 
                        if buffer[pos][3]:
                            endPos = pos
                            break

                    sendBuffer.append((buffer[x][0],buffer[endPos][1],rewardTraj,buffer[x][3],buffer[x][4]))

                    if buffer[endPos][3]:
                        tdError = rewardTraj - qs[x]
                    else:
                        tdError = rewardTraj- qs[x] + GAMMA * qs[endPos]

                    prios.append(abs(tdError))

                queue.put((sendBuffer, prios))
                qs = qs[len(sendBuffer):]
                buffer = buffer[len(sendBuffer):]

        localGameNumber += 1

        with stats.gameNumber.get_lock():
            stats.gameNumber.value += 1

        if score > stats.bestScore.value:
            stats.bestScore.value = score   

        vis.line(X=[localGameNumber-1], Y=[score], win=name2, opts=dict(title=name2), update="append")                 

class Stats:
    def __init__(self):
        self.frameNumber = mp.Value("i", 0)
        self.gameNumber = mp.Value("i", 0)
        self.batchNumber = mp.Value("i", 0)
        self.bestScore = mp.Value("d", 0)
        self.loss = mp.Value("d", 0)
        self.rndLoss = mp.Value("d", 0)

def main():   
    Q = DQNet(NUM_ACTIONS)#.to(device)  #main
    target = Predictor()
    predictor = Predictor()
    
    if RESUME_TRAINING:
        files = os.listdir(MODEL_SAVE_FOLDER)
        files.sort()
        lastFile = files[len(files)-1]
        checkpoint = torch.load(os.path.join(MODEL_SAVE_FOLDER, lastFile))
        Q.load_state_dict(checkpoint["model"])
        target.load_state_dict(checkpoint["target"])
        predictor.load_state_dict(checkpoint["pred"])
        print("loaded from lastest checkpoint with name: " + str(lastFile))

    Q.share_memory()
    threads = max(2,NUM_THREADS) # needs atleast 2 threads
    stats = Stats()

    processes = []
    man = mp.Manager()
    queue = man.Queue(50)

    #a variety of epsilsons values picked to be in range 0.01->0.5, skewed to the left
    #epsilons = [0.01, 0.5, 0.1, 0.01, 0.12, 0.05, 0.2, 0.1, 0.3, 0.2, 0.14, 0.02, 0.25] 
    epsilons = [0.1 for _ in range(30)] 
    for x in range(threads-1):
        p = mp.Process(target=actor, args=(Q, target, predictor, stats, queue,x+1,epsilons[x]),) #epsilon varies from 0 to 0.5
        processes.append(p)
        p.start()

    p = mp.Process(target=learner, args=(Q, queue, stats, target, predictor),)
    p.start()
    processes.append(p)    

    for each in processes:
        each.join()

if __name__ == "__main__":
    main()

   # test()

