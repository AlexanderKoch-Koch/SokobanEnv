import numpy as np


class Experience:

    def __init__(self, maxlen):
        self.memory = np.zeros(shape=[maxlen, 6], dtype=object)
        self.index = 0
        self.maxlen = maxlen
        self.len = 0

    def store(self, state, action, reward, state_next, done, priority):
        a = np.arange(6)
        b = np.array([state, action, reward, state_next, done, priority])

        self.memory[self.index] = b
        self.index = (self.index + 1) % self.maxlen
        self.len = min(self.len + 1, self.maxlen)

    def sample(self, prioritized_replay=False):
        if prioritized_replay:
            prob = np.array(self.memory[:, 5], dtype='float64')
            prob /= prob.sum()
            index_chosen = np.random.choice(self.maxlen, p=prob)
        else:
            index_chosen = np.random.choice(self.len)
        return self.memory[index_chosen], index_chosen

    def update_td_error(self, index, td_error):
        self.memory[index, 5] = td_error
