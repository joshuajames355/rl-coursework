import gym
import numpy as np
import random
import visdom

vis = visdom.Visdom(port=12345)

env = gym.make("FrozenLake-v0", is_slippery=False, map_name="8x8")

env.render()

EPSILON = 0.9
GAMMA = 0.9
Q = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.1

NUM_EPISODES = 50000

BATCH_SIZE = 200
batch_score = 0
averageScore = []
greedyScore = []

def epsilon_greedy(q,obs, epsilon):
    if random.random() > epsilon:
        return np.argmax(q[obs])
    return env.action_space.sample()

def greedy(q,obs):
    return np.argmax(q[obs])

def state_value(q, state):
    return max(q[state])

for batch_num in range(NUM_EPISODES):
    e = min(1,2000/(batch_num + 1))

    g = 0 # total reward
    obs =  env.reset()
    done = False
    while not done:
        action = epsilon_greedy(Q,obs,e)
        oldObs = obs
        obs,reward,done,inf = env.step(action)
        Q[oldObs,action] = Q[oldObs,action] + alpha * (reward + GAMMA * state_value(Q, obs) - Q[oldObs,action])
        g += reward

    if batch_num % BATCH_SIZE == 0 and batch_num != 0:
        averageScore.append(batch_score)
        vis.line(averageScore, win="Score")
        batch_score = 0

    batch_score += g/BATCH_SIZE

print(Q)
