# basic implementation of a differential evolution algorithm
# @tz2lala

from optbase import OptimizerPopulationBased
from typing import List, Tuple
import numpy as np
from testfuns import EGG_HOLDER

class DifferentialEvolution(OptimizerPopulationBased):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def optimize(self, crossover_rate: float,
                 differential_weight: float,
                 max_gen: float = 10):

        self.crossover_rate = crossover_rate
        self.differential_weight = differential_weight
        self._cost_trace = np.zeros(max_gen)
        # Initial population
        self._pop = self.x0
        func_val = self.evaluate_population(self._pop)
        self._cost_trace[0] = func_val.min()
        
        # Perform optimization iterations
        for generation in range(max_gen):
            # Create offspring population
            new_population = self.create_offspring_population()

            # Evaluate offspring population
            new_func_val = self.evaluate_population(new_population)

            # Select individuals for the next generation
            update_ids = new_func_val < func_val
            self._pop[update_ids] = new_population[update_ids]

            # record the best cost
            func_val[update_ids] = new_func_val[update_ids]
            self._cost_trace[generation] = func_val.min()

        # Return the best individual found
        self._x_est = self._pop[func_val.argmin()]


    def evaluate_population(self, population) -> np.ndarray:
        # evaluate the function values for the population
        func_val = np.zeros(population.shape[0])
        for i in range(population.shape[0]):
            func_val[i] = self.objective_function(population[i])
        return func_val
        

    def create_offspring_population(self):
        # generate a new generation of population

        # prepare bounds for clipping
        lower = np.array([pair[0] for pair in self.bounds])
        higher = np.array([pair[1] for pair in self.bounds])

        # create new population
        new_pop = np.zeros_like(self._pop)
        for this_id, this_member in enumerate(self._pop):
            # select 3 random members that is not this member
            candidate_ids = np.arange(self._pop.shape[0])
            candidate_ids = np.delete(candidate_ids, this_id)
            chosen_ids = np.random.choice(candidate_ids, 3, replace=False)

            # interim member
            interim_member = self._pop[chosen_ids[0]] + \
                self.differential_weight * (self._pop[chosen_ids[1]] - self._pop[chosen_ids[2]])
            
            # choose dimension of crossover
            crossover_dims = np.random.choice(
                np.arange(self._pop.shape[1]))
            
            # crossover
            new_member = np.zeros_like(this_member)
            for dim_id in range(this_member.size):
                if np.random.rand() < self.crossover_rate or dim_id == crossover_dims:
                    new_member[dim_id] = interim_member[dim_id]
                else:
                    new_member[dim_id] = this_member[dim_id]
            
            # clipping
            new_member = np.clip(new_member, lower, higher)

            new_pop[this_id] = new_member

        return new_pop

if __name__ == '__main__':
    # Run an example
    de = DifferentialEvolution(objective_function=EGG_HOLDER['obj'],
                               bounds=EGG_HOLDER['bounds'])
    de.initialize_population(30)
    de.optimize(crossover_rate=0.5, differential_weight=0.5, max_gen=100)
    de.plot_trace()