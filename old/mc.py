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
N = np.zeros([env.observation_space.n, env.action_space.n])

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


for batch_num in range(NUM_EPISODES):
    e = min(1,2000/(batch_num + 1))

    g = 0
    obs =  env.reset()
    done = False
    path = []
    rewards = []
    while not done:
        action = epsilon_greedy(Q,obs,e)
        path.append((obs, action))
        obs,reward,done,inf = env.step(action)
        rewards.append(reward)

    g = 0
    for x in range(len(path)-1, -1, -1):
        g = g*GAMMA + rewards[x]
        state = path[x]
        N[state[0], state[1]] +=1
        Q[state] = Q[state] + (1/N[state]) * (g - Q[state])

    if batch_num % BATCH_SIZE == 0 and batch_num != 0:
        averageScore.append(batch_score)
        vis.line(averageScore, win="Score")
        batch_score = 0

    batch_score += g/BATCH_SIZE

print(Q)
