import torch
import torch.nn as nn
import gym

env = gym.make("Gravitar-v0")
print(env.unwrapped.get_action_meanings())

env = gym.make("Pong-v0")
print(env.unwrapped.get_action_meanings())

file = "dqnModel/2021-01-15_23-49-02_104"

device = "cpu"

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, env.action_space.n),
            nn.Softmax()
        )

    def forward(self, state):
        return self.network(state)

    def action_value(self, state, action): #q(s, a)
        answer = self(torch.tensor(state, dtype=torch.float32).to(device))
        return answer[action]

    def max_action_value(self, state): # max_a q(s,a) , returns q(s,a), a
        answer = self(torch.tensor(state, dtype=torch.float32).to(device))
        arr = answer.cpu().detach()
        action = np.argmax(arr)
        return (arr[action], action) 

def printStuff(model):
    for x in model.parameters():
        print(x)

    obs = env.reset()

    print()
    print(obs)
    print()
    print(model(torch.tensor(obs, dtype=torch.float32)))

def runEval(model, env):
    #env = gym.wrappers.Monitor(env, 'videos', force = True)

    runTimes = 50
    avg = 0

    for _ in range(runTimes):
        obs = env.reset()
        done = False
        g = 0
        while not done:
            action = model(torch.tensor(obs, dtype=torch.float32)).argmax().item()
            obs,reward,done,inf = env.step(action)

            g += reward
        avg += float(g)/float(runTimes)
    print(avg)

model = torch.load(file)
runEval(model, env)