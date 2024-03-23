# Basic implementation of a particle swarm optimization algorithm
# @tz2lala

import numpy as np
from typing import List, Tuple
from optbase import OptimizerBase


class PSO(OptimizerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_population(self, population_size: int, 
                              bounds: List[Tuple]) -> None:
        # Generate initial population with uniform random distribution
        self.n_pop = population_size
        self.n_var = len(bounds)
        if bounds is not None: self.bounds = bounds
        lower = np.array([pair[0] for pair in self.bounds])
        higher = np.array([pair[1] for pair in self.bounds])
        self.x0 = np.random.uniform(lower, higher, (self.n_pop, self.n_var))



    def optimize(self, max_gen: int,
                 c1: float = 0.1, c2: float = 0.1, inertia: float = 1.0,
                 init_velocity_scale: float = 0.1):
        if self.n_pop is None:
            raise ValueError("Population not initialized")

        self._cost_trace = np.zeros(max_gen)

        # initialize best position storage
        self._pop = self.x0
        self.evaluate_best()

        # initialize velocity
        velocity = np.random.randn((self.n_pop, self.n_var)) * init_velocity_scale

        # iterations
        for igen in range(max_gen):
            # update velocity
            r1 = np.random.rand(self.n_pop, self.n_var) # random scalings
            r2 = np.random.rand(self.n_pop, self.n_var)
            velocity = inertia * velocity + \
                c1 * r1 * (self.best_pos_p - self.x0) + \
                c2 * r2 * (self.best_pos_g - self.x0)

            # update position
            self.x0 += velocity

            # update best positions
            self.evaluate_best()

            self._cost_trace[igen] = self.best_val_g
            self._x_est = self.best_pos_g


    def evaluate_best(self):
        if self.best_pos_p is None:
            self.best_pos_p = self._pop
            self.best_val_p = self.evaluate_population(self._pop)
            self.best_pos_g = self.best_val_p[self.best_val_p.argmin()]
            self.best_val_g = self.best_val_p.min()
        else:
            func_val = self.evaluate_population(self._pop)

            # update particle best
            update_ids = func_val < self.best_val_p
            self.best_val_p[update_ids] = func_val[update_ids]
            self.best_pos_p[update_ids] = self._pop[update_ids]

            # update global best
            if self.best_val_p.min() < self.best_val_g:
                self.best_val_g = self.best_val_p.min()
                self.best_pos_g = self.best_pos_p[self.best_val_p.argmin()]



    
    def evaluate_population(self, population) -> np.ndarray:
        # evaluate the function values for the population
        func_val = np.zeros(population.shape[0])
        for i in range(population.shape[0]):
            func_val[i] = self.objective_function(population[i])
        return func_val