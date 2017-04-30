#! /usr/bin/env python

"""
Contains the class Population and it's relevant methods.
"""

from __future__ import division

__author__ = "Padraig O Neill"

import random

import genome


class Population:
    """Population class"""

    mutation_rate = 0.0
    genomes = []
    mating_pool = []
    generations = 0
    finished = False
    elitism = 0
    culling_rate = 0
    random_spawn_rate = 0
    genome_size =0
    action_range =0


    def __init__(self, m, population_size, genome_size, action_range, elit,spawn_rate,cull_rate):
        """Return a DNA object """
        self.mutation_rate = m
        for i in range(population_size):
            self.genomes.append(genome.Genome(genome_size, action_range))

        self.random_spawn_rate = spawn_rate
        self.genome_size = genome_size
        self.action_range= action_range
        self.generations = 0
        self.elitism = elit
        self.culling_rate = cull_rate


    def calc_fitness(self,individual,fitness):
        self.genomes[individual].set_fitness(fitness)

    def natural_selection(self):
        del self.mating_pool[:]
        self.rescale_fitness()
        culled = self.get_n_worst(int(len(self.genomes)*self.culling_rate))
        for g in self.genomes:
            if g not in culled:
                n = int(g.get_fitness() + 1)
                for j in range(0, n, 1):
                    self.mating_pool.append(g)

    def generate(self):
        elite = self.get_n_best(self.elitism)
        count = 0
        for g in range(len(self.genomes) - self.elitism,len(self.genomes),1):
            self.genomes[g] = elite[count]
            count +=1

        for g in range(int(self.random_spawn_rate*len(self.genomes)),len(self.genomes)-self.elitism):
            a = random.randint(0, len(self.mating_pool) - 1)
            b = random.randint(0, len(self.mating_pool) - 1)
            parent_a = self.mating_pool[a]
            parent_b = self.mating_pool[b]

            child = parent_a.crossover(parent_b)
            child.mutate(self.mutation_rate)
            self.genomes[g] = child


        for g in range(int(self.random_spawn_rate*len(self.genomes))):
            self.genomes[g] = genome.Genome(self.genome_size, self.action_range)
        self.generations += 1


    def get_best(self):
        best_fitness_score = 0.0
        best_genome = None
        for genome in self.genomes:
            if genome.get_fitness() > best_fitness_score:
                best_genome = genome
                best_fitness_score = genome.get_fitness()

        return best_genome


    def get_worst(self):
        # High worst score to work from
        worst_fitness_score = 10000000000000.0
        worst_genome = None
        for genome in self.genomes:
            if genome.get_fitness() < worst_fitness_score:
                worst_genome = genome
                worst_fitness_score = genome.get_fitness()

        return worst_genome


    def get_n_best(self, n):
        sorted_list =sorted(self.genomes, key=lambda x: x.get_fitness(), reverse=True)
        return sorted_list[:n]

    def get_n_worst(self, n):
        sorted_list =sorted(self.genomes, key=lambda x: x.get_fitness(), reverse=True)
        return sorted_list[-n:]


    def get_finished(self):
        return self.finished

    def get_generations(self):
        return self.generations

    def get_average_fitness(self):
        total = 0
        for individual in self.genomes:
            total += individual.fitness
        return total / len(self.genomes)

    def all_genotypes(self):
        everything = ""
        display_limit = min(len(self.genomes), 50)

        for i in range(display_limit):
            everything += self.genomes[i].get_moves() + "\n"
        return everything


    def rescale_fitness(self, new_min=0, new_max=1):
        old_min, old_max = min(self.genomes, key=lambda x: x.get_fitness()), \
                           max(self.genomes, key=lambda x: x.get_fitness())

        for g in self.genomes:
            g.fitness_scale = (new_max - new_min) / (old_max.get_fitness() - old_min.get_fitness()) * \
                              (g.get_fitness() - old_min.get_fitness()) + new_min


