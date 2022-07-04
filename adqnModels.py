
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