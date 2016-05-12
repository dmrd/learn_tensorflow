import math
import numpy as np


class MLP(object):
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.weights = []
        for (i, dim) in enumerate(layer_dims[1:]):
            self.weights.append(np.zeros((layer_dims[i], dim)))
        self.nparams = sum(x.size for x in self.weights)

    def forward(self, x):
        for W in self.weights:
            x = np.dot(x, W)
        return x

    def get_flattened(self):
        ret = np.zeros(self.nparams)
        start = 0
        for W in self.weights:
            sz = W.size
            ret[start:start + sz] = W.flat
            start += sz
        return ret

    def set_flattened(self, params):
        assert params.size == self.nparams
        start = 0
        for W in self.weights:
            sz = W.size
            W.flat = params[start:start + sz]
            start += sz


class Agent(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
    def step(state):
        pass


def simulate(env, agent, steps):
    obs = env.reset()
    for _ in range(steps):
        action = agent.step(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break
    return reward
