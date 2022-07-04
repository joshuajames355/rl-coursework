import gym

import visdom
import torch
from utils import *
from models import *

vis = visdom.Visdom(port=12345, env="Preprocessing Test")

env = gym.make("Breakout-v0")
obs = env.reset()

print(env.unwrapped.get_action_meanings())
print(env.action_space.n)

obs,reward,done,inf = env.step(env.action_space.sample())

#print(torch.tensor(obs).permute(2,0,1).size())

#vis.image(obs)

#q = R2D2Net()

#out = torch.tensor(obs).permute().numpy()

grayscale = preprocess(obs)
out2 = grayscale.numpy()

vis.image(obs.transpose(2,0,1))
#vis.image(out2)

q(grayscale.unsqueeze(0), q.getNewHistory())