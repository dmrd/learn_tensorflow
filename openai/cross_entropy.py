import sys
import math
import numpy as np
import gym

if len(sys.argv) > 1:
    ENV = sys.argv[1]
    if len(sys.argv) == 3:
        exp_name = sys.argv[2]
else:
    ENV = "CartPole-v0"
    exp_name = None

class MLPAgent(object):
    def __init__(self, layer_dims, ):
        self.layer_dims = layer_dims
        self.weights = []
        for (i, dim) in enumerate(layer_dims[1:]):
            self.weights.append(np.zeros((layer_dims[i], dim)))
        self.nparams = sum(x.size for x in self.weights)

    def forward(self, x):
        for W in self.weights:
            x = np.tanh(np.dot(x, W))
        return x

    """
    Get the params as a single vector.  Concatenated weight matrices
    """
    def get_flattened(self):
        ret = np.zeros(self.nparams)
        start = 0
        for W in self.weights:
            sz = W.size
            ret[start:start + sz] = W.flat
            start += sz
        return ret

    """
    Get the weight matrices from a single flattened vector of params.
    Inverse of get_flattened
    """
    def set_flattened(self, params):
        assert params.size == self.nparams
        start = 0
        for W in self.weights:
            sz = W.size
            W.flat = params[start:start + sz]
            start += sz

    """
    Return the action given state
    Just argmax over possible actions for now (i.e. no stochastic sampling)
    """
    def step(self, state):
        y = self.forward(state)
        return np.argmax(y)


def simulate(env, agent, max_steps, render=False):
    obs = env.reset()
    total_reward = 0.0
    for _ in xrange(max_steps):
        action = agent.step(obs)
        obs, reward, done, info = env.step(action)
        if render:
            env.render()
        total_reward += reward
        if done:
            break
    return total_reward


def eval_agent(env, agent, max_steps):
    def eval_params(params, render=False):
        agent.set_flattened(params)
        return simulate(env, agent, max_steps, render=render)
    return eval_params

def cem(f, iters=10, n_samples=100, keep_percent=0.2, std_noise=0.0, noise_decay=1):
    means = np.zeros(agent.nparams)
    stds = np.ones(agent.nparams)

    n_elite = int(round(keep_percent * n_samples))
    for generation in xrange(iters):
        stds += std_noise * pow(noise_decay, generation)
        samples = np.random.multivariate_normal(means, np.diag(stds), n_samples)
        rewards = np.zeros(n_samples)
        for i, sample in enumerate(samples):
            rewards[i] = f(sample, render=(i == 0))
        best_idx = rewards.argsort()[-n_elite:]
        elite = samples[best_idx]
        means = elite.mean(axis=0)
        stds = elite.var(axis=0)

        print("{}: Average reward: {:10} | elite reward: {:10}".format(
            generation + 1, rewards.mean(), rewards[best_idx].mean()))

env = gym.make(ENV)
agent = MLPAgent([sum(env.observation_space.shape), 10, 5, env.action_space.n])
if exp_name:
    env.monitor.start(exp_name)
cem(eval_agent(env, agent, 200), iters=30)
if exp_name:
    env.monitor.close()
