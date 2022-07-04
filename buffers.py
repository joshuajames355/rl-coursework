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


class ReplayBuffer():
    def __init__(self, maxSize = 1000000):
        self._maxSize = maxSize
        self._data = []

    def size(self):
        return len(self._data)

    def push(self, elem):
        self._data.append(elem)
        if len(self._data) > self._maxSize:
            del self._data[0]

    def getElem(self):
        return random.choice(self._data)

class PriorityReplayBufferTensor():
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

        print(self.size())

        if self.size() >= self.maxSize * 1.2:
            buffer.makeRoom()

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
