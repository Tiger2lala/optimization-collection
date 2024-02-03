# genetic optimizer, basic implementation
# @tz2lala

from optbase import OptimizerBase
from typing import Tuple
import numpy as np
from testfuns import MY_QUARTIC

class GeneticOptimizer(OptimizerBase):
    def __init__(self, 
                 mutation_rate: float = 0.01,
                 fitness_temperature: float = 1.,
                 **kwargs):
        super().__init__(**kwargs)
        self.mutation_rate = mutation_rate
        self.fitness_temperature = fitness_temperature
        self._pop = None


    def initialize_population(self, population_size: int = 10, 
                              n_bit: int = 10):
        """
        Generate initial population of binary strings.
        Default uniform in all digits.
        """
        if population_size: self._n_pop = population_size
        if n_bit: self._n_bit = n_bit
        self.x0 = np.random.randint(0,2,(population_size,n_bit))


    def _evaluate_fitness(self):
        funcval = np.zeros(self._n_pop)
        for j in range(self._n_pop):
            x = self._decode(self._pop[j], self.bounds[0])
            funcval[j] = self.objective_function(x)
        fitness = self._calc_fitness(funcval, self.fitness_temperature)
        return fitness

    @staticmethod
    def _calc_fitness(funcval: np.ndarray, temperature: float) -> np.ndarray:
        return np.exp(-funcval / temperature)

    def _decode(self, bits: np.ndarray, bounds: Tuple) -> float:
        # Decoding bit string into the variable
        # The bits (base 1/2) translates to a float in [0,1]
        bitvar = 0
        for i, bit_i in enumerate(bits):
            bitvar += (1/2)**(i+1) * bit_i
        
        flvar = bounds[0] + \
            (bounds[1] - bounds[0]) * bitvar
        return flvar

    @staticmethod
    def _select_parents(pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        # select 2 genes from the population based on fitness
        # using roulette wheel selection
        # need fitness to be normalized
        chosen_indices = np.random.choice(np.arange(0,fitness.size),
                                        2,
                                        replace=False,
                                        p=fitness)
        return pop[chosen_indices]

    @staticmethod
    def _crossover(gene1: np.ndarray, gene2: np.ndarray) -> \
        Tuple[np.ndarray]:
        # crossover 2 genes at 1 random place
        assert gene1.size == gene2.size
        cross_point = np.random.randint(1, gene1.size-1)
        outgene1 = gene1
        outgene2 = gene2
        outgene1[cross_point:] = gene2[cross_point:]
        outgene2[:cross_point] = gene1[:cross_point]
        return outgene1, outgene2

    @staticmethod
    def _mutate(gene: np.ndarray, prob: float) -> np.ndarray:
        # mutate the gene according to probability
        mut_mask = np.random.binomial(1, prob, gene.size)
        return np.bitwise_xor(gene, mut_mask)

    def optimize(self, niter: int = None,
                 mutation_rate: float = None):
        if niter: self._niter = niter
        if mutation_rate: self.mutation_rate = mutation_rate
        self._cost_trace = np.zeros(self._niter)
        best_fit = 0

        self._pop = self.x0
        # iter over generations
        for i in range(self._niter):
            # evaluate fitness for each individual
            fitness = self._evaluate_fitness()
            # get best individual
            best_index = np.argmax(fitness)
            # update best
            if fitness[best_index] > best_fit:
                best_fit = fitness[best_index]
                best_gene = self._pop[best_index]
            self._cost_trace[i] = \
                self.objective_function(self._decode(best_gene, self.bounds[0]))
            
            # create new generation
            new_pop = np.zeros((self._n_pop, self._n_bit), dtype=int)
            for j in range(int(self._n_pop/2)): # loop for each pair
                chosen_genes = self._select_parents(self._pop, fitness/np.sum(fitness))
                new_gene1, new_gene2 = self._crossover(chosen_genes[0], chosen_genes[1])
                new_pop[j*2] = self._mutate(new_gene1, self.mutation_rate)
                new_pop[j*2+1] = self._mutate(new_gene2, self.mutation_rate)

            self._pop = new_pop
        return self._cost_trace[-1]


if "__name__" == "__main__":
    # Run an example
    options = {'objective_function': MY_QUARTIC['obj'],
                'bounds': MY_QUARTIC['bounds'],
                'mutation_rate': 0.1,
                'fitness_temperature': 2.}
    ga = GeneticOptimizer(**options)
    ga.initialize_population(population_size=10, n_bit=16)
    ga.optimize(niter=50)
    ga.plot_trace()