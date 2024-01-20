# An attempt of genetic algorithm
# @tz2lala

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple

# simple example objective function
def objective(x: float) -> float:
    # in [-3,3] the min is at (-2.24, -12.9)
    return x**4 + x**3 - 7*x**2 - x + 6


def calc_fitness(funcval: np.ndarray, temperature: float) -> np.ndarray:
    # normalized fitness function
    fitness = np.exp(-funcval / temperature)
    return fitness


def decode(bits: np.ndarray, bounds: tuple) -> float:
    # Decoding bit string into the variable
    n_bits = len(bits)
    
    # The bits (base 1/2) translates to a float in [0,1]
    bitvar = 0
    for i, bit_i in enumerate(bits):
        bitvar += (1/2)**(i+1) * bit_i
    
    flvar = bounds[0] + (bounds[1] - bounds[0]) * bitvar
    return flvar


def crossover(gene1: np.ndarray, gene2: np.ndarray) -> \
    Tuple[np.ndarray]:
    # crossover 2 genes at 1 random place
    assert gene1.size == gene2.size
    cross_point = np.random.randint(1, gene1.size-1)
    outgene1 = gene1
    outgene2 = gene2
    outgene1[cross_point:] = gene2[cross_point:]
    outgene2[:cross_point] = gene1[:cross_point]
    return outgene1, outgene2


def mutation(gene: np.ndarray, prob: float) -> np.ndarray:
    # mutate the gene according to probability
    mut_mask = np.random.binomial(1, prob, gene.size)
    return np.bitwise_xor(gene, mut_mask)


def selection(pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    # select 2 genes from the population based on fitness
    # using roulette wheel selection
    # need fitness to be normalized
    chosen_indices = np.random.choice(np.arange(0,fitness.size),
                                      2,
                                      replace=False,
                                      p=fitness)
    return pop[chosen_indices]


def ga_main(targ: Callable, val_range: Tuple, n_bit: int) -> float:
    """
    Main function for the genetic algorithm.

    Args:
        targ (function): The target function to be optimized.
        val_range (Tuple): The range of values for the variables in the target function.
        n_bit (int): The number of bits used to represent each variable.

    Returns:
        float: The optimized value of the target function.
    """

    # Initialize the population
    n_pop = 10
    pop = np.random.randint(0,2,(n_pop,n_bit))
    p_mut = 0.1
    
    # iterations
    n_iter = 50
    best_gene = np.zeros(n_bit)
    best_fit = 0
    temperature = 1.
    func_trace = np.zeros(n_iter)
    for i in range(n_iter):
        # Evaluate the fitness
        funcval = np.zeros(n_pop)
        for j in range(n_pop):
            x = decode(pop[j], val_range)
            funcval[j] = targ(x)
        fitness = calc_fitness(funcval, temperature)

        # Find the best gene
        best_index = np.argmax(fitness)
        if fitness[best_index] > best_fit:
            best_fit = fitness[best_index]
            best_gene = pop[best_index]
        func_trace[i] = targ(decode(best_gene, val_range))

        # Selection
        new_pop = np.zeros((n_pop, n_bit), dtype=int)
        for j in range(int(n_pop/2)): # loop for each pair
            chosen_genes = selection(pop, fitness/np.sum(fitness))
            new_gene1, new_gene2 = crossover(chosen_genes[0], chosen_genes[1])
            new_pop[j*2] = mutation(new_gene1, p_mut)
            new_pop[j*2+1] = mutation(new_gene2, p_mut)
        
        pop = new_pop

    fig, ax = plt.subplots()
    ax.plot(func_trace)
    fig.show()
    return func_trace[-1]


if __name__ == "__main__":
    print(ga_main(objective, (-3,3), 16))
