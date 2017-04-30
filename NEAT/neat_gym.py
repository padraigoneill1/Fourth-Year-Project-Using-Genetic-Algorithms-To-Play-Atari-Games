# ! /usr/bin/env python

"""
This script is responsible for running the NEAT- Python implementation,
in using OpenAI Gym for evaluation.
"""

__author__ = "Padraig O Neill"

import argparse
import os

import gym
import numpy as np

import neat
import visualize

# Set the game to be tested here
game_name = 'AirRaid-ram-v0'

# Games Tested
# 'Breakout-ram-v0'
# 'AirRaid-ram-v0'
# 'SpaceInvaders-ram-v0'
# 'MsPacman-ram-v0'
# 'Asterix-ram-v0'


parser = argparse.ArgumentParser(description='OpenAI NEAT Simulator')
parser.add_argument('--max-steps', dest='max_steps', type=int, default=10000,
                    help='The max number of steps to take per episode')
parser.add_argument('--episodes', type=int, default=1,
                    help="Number of episodes to run each genome for")
parser.add_argument('--render', action='store_true')
parser.add_argument('--generations', type=int, default=150,
                    help="Number of generations to evolve for")
parser.add_argument('--checkpoint', type=str,
                    help="Start from a previous checkpoint. Specify the checkpoint file name")
parser.add_argument('--num-cores', dest="numCores", type=int, default=4,
                    help="Python- neat allows for parallel execution of code. Specify amount of cores available")
args = parser.parse_args()


def evaluate_fitness(net, env, episodes=1, steps=5000, render=False):
    # Specify seed below if the same seed is required for every game.
    # env.seed(0)  Have same seed for all game
    fitnesses = []
    for episode in range(episodes):
        inputs = game_env.reset()  # Receive observation from OpenAI as Inputs
        total_reward = 0.0

        for j in range(steps):
            outputs = net.activate(inputs)
            action = np.argmax(outputs)
            inputs, reward, done, info = env.step(action)
            if render:
                env.render()
            if done:
                print("Episode finished after {} steps".format(j + 1))
                break
            total_reward += reward

        fitnesses.append(total_reward)

    fitness = np.array(fitnesses).mean()
    print("Genome fitness: %s" % str(fitness))
    return fitness


def evaluate_genome(g, conf):
    """ Send a genome & config file to the neat implementation and receive its phenotype (a FeedForwardNetwork). """
    net = neat.nn.FeedForwardNetwork.create(g, conf)
    return evaluate_fitness(net, game_env, args.episodes, args.max_steps, render=args.render)


def run_neat(env):
    def eval_genomes(genomes, conf):
        for g in genomes:
            fitness = evaluate_genome(g, conf)
            g.fitness = fitness

    # Locate config file & Load
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'openAI_neat_config')
    conf = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                              neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create population from config file
    pop = neat.population.Population(conf)

    # Create Statistics reporter
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Create automatic checkpointer to create checkpoints
    pop.add_reporter(neat.Checkpointer(10, 900))

    # Load checkpoint if passed in
    # Checkpoint loading not working
    # if args.checkpoint:
    #   neat.Checkpointer.restore_checkpoint(args.checkpoint)

    # Start simulation with render. No Parallel Processing not available if game is being rendered.
    if args.render:
        pop.run(eval_genomes, args.generations)  # Provide fitness function & generations to run
    else:
        parallel_evaluator = neat.parallel.ParallelEvaluator(args.numCores, evaluate_genome)
        pop.run(parallel_evaluator.evaluate, args.generations)

    stats.save()

    # Show output of the current most fit genome against training data.
    best_genome = pop.best_genome

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(best_genome))
    # Show output of the most fit genome against training data.
    print('\nOutput:')

    visualize.draw_net(conf, best_genome, True, filename="winner_net.svg")
    visualize.plot_stats(stats, ylog=False, view=False, filename="stats.svg")
    visualize.plot_species(stats, view=True)

    # Give option to run fittest individual again
    raw_input("Press Enter to run the best genome...")
    best_genome_net = neat.nn.FeedForwardNetwork.create(best_genome, conf)
    for i in range(100):
        evaluate_fitness(best_genome_net, env, 1, args.max_steps, render=True)

    env.close()


game_env = gym.make(game_name)
print ("Input Nodes: %s" % str(len(game_env.observation_space.high)))
print ("Output Nodes: %s" % str(game_env.action_space.n))
print("Action space: {0!r}".format(game_env.action_space))
print("Observation space: {0!r}".format(game_env.observation_space))
run_neat(game_env)
