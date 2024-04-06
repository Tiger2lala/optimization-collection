# Basic implementation of a particle swarm optimization algorithm
# @tz2lala

import numpy as np
from typing import List, Tuple
from optbase import OptimizerPopulationBased
from testfuns import EGG_HOLDER


class PSO(OptimizerPopulationBased):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_pos_p = None
        self.best_val_p = None
        self.best_pos_g = None
        self.best_val_g = None


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
        velocity = np.random.randn(self.n_pop, self.n_var) * init_velocity_scale

        # iterations
        for igen in range(max_gen):
            # update velocity
            r1 = np.random.rand(self.n_pop, self.n_var) # random scalings
            r2 = np.random.rand(self.n_pop, self.n_var)
            velocity = inertia * velocity + \
                c1 * r1 * (self.best_pos_p - self._pop) + \
                c2 * r2 * (self.best_pos_g - self._pop)

            # update position
            self._pop += velocity

            # clipping
            lower = np.array([pair[0] for pair in self.bounds])
            higher = np.array([pair[1] for pair in self.bounds])
            self._pop = np.clip(self._pop, lower, higher)

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


if __name__ == '__main__':
    # Run an example
    de = PSO(objective_function=EGG_HOLDER['obj'],
             bounds=EGG_HOLDER['bounds'])
    de.initialize_population(30)
    de.optimize(inertia=0.8, max_gen=100)
    de.plot_trace()