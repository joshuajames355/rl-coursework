import torch.nn as nn
import torch

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.ReLU(),
        )

    def forward(self, state):
        return self.network(state)

#dueling qnet
class DQNet(nn.Module):
    def __init__(self,  numActions = 2):
        super(DQNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
        )

        self.valueStream = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

        self.advantageStream = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, numActions),
        )

    def forward(self, state):
        x = self.network(state)
        value = self.valueStream(x)
        advantage = self.advantageStream(x)
        return value + (advantage - advantage.mean())

#dueling qnet
class R2D2Net(nn.Module):
    def __init__(self, actionSize=18):
        super(R2D2Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1,4,kernel_size=8,stride=4),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4,16,kernel_size=4,stride=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),#16*9 output
        )

        self.linear = nn.Linear(16*5*5, 128)       

        self.ltsm = nn.LSTM(128,128,1)

        self.valueStream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.advantageStream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, actionSize), #need to check output size
        )

    def getNewHistory(self):
        return (torch.zeros((1, 1, 128)), torch.zeros((1, 1, 128)))

    def forward(self, state, history, train=False):
        x = self.conv(state)
        x = self.linear(torch.flatten(x, 1))
        x, history = self.ltsm(x.unsqueeze(1).permute(1, 0 , 2), history)
        x = x.squeeze(0)
        value = self.valueStream(x)
        advantage = self.advantageStream(x)
        return (value + (advantage - advantage.mean()), history)