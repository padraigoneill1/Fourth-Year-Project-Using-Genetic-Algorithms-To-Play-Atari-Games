# ! /usr/bin/env python

"""
Runs a random agent for 1000 generations, to find an average fitness value environment.
"""

__author__ = "Padraig O Neill"


import argparse

import gym
import numpy as np

parser = argparse.ArgumentParser(description='OpenAI Random Player Simulation')
parser.add_argument('--render', action='store_true', default='True')
parser.add_argument('env_id', nargs='?', default='MsPacman-ram-v0', help='Select the environment to run')
args = parser.parse_args()

players = []

env = gym.make(args.env_id)
env.seed(0)

print("Input Nodes: %s" % str(len(env.observation_space.high)))
print("Output Nodes: %s" % str(env.action_space.n))
print("Action space: {0!r}".format(env.action_space))
print("Observation space: {0!r}".format(env.observation_space))

for i in range(1):
    print("Individual No: %s" % str(i))
    fitnesses = []
    observation = env.reset()
    total_reward = 0.0
    for t in range(5000):
        if args.render:
            env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if reward != 0.0:
            print(reward)
            # time.sleep(5.5)
        if done:
            print("Episode finished after {} steps".format(t + 1))
            break

        total_reward += reward
        fitnesses.append(total_reward)

    fitness = np.array(fitnesses).mean()
    players.append(fitness)
    print("Random Agent fitness: %s" % str(fitness))

print("Average fitness over all: %s" % str(np.array(players).mean()))
