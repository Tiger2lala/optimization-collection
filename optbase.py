# base class for my rudimentary implementation of optimizers
# class structure to help compare algorithms
# @tz2lala

from typing import Callable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

class OptimizerBase:
    def __init__(self, 
                 objective_function: Callable = None,
                 gradient: Callable = None,
                 bounds: List[Tuple] = None,
                 x0: np.ndarray = None):
        self.objective_function = objective_function
        self.bounds = bounds
        self.x0 = x0
        self._cost_trace = None
        self._x_est = None
        self._converged = False
        self.niter = 1
        self.grad = gradient
        
    def get_cost_trace(self):
        return self._cost_trace
    
    def get_x_est(self):
        return self._x_est
    
    def get_converged(self):
        return self._converged

    def optimize(self):
        raise NotImplementedError("Subclasses must implement the optimize method.")
    
    def plot_trace(self):
        fig, ax = plt.subplots()
        ax.plot(self._cost_trace)
        fig.show()
        return fig, ax
    

class OptimizerPopulationBased(OptimizerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pop = None
        self.n_pop = None
        self.n_var = None
        
    
    def initialize_population(self, population_size: int, 
                              bounds: List[Tuple] = None) -> None:
        # Generate initial population with uniform random distribution
        if bounds is not None: self.bounds = bounds
        self.n_pop = population_size
        self.n_var = len(self.bounds)
        lower = np.array([pair[0] for pair in self.bounds])
        higher = np.array([pair[1] for pair in self.bounds])
        self.x0 = np.random.uniform(lower, higher, (self.n_pop, self.n_var))


    def evaluate_population(self, population) -> np.ndarray:
        # evaluate the function values for the population
        func_val = np.zeros(self.n_pop)
        for i in range(self.n_pop):
            func_val[i] = self.objective_function(population[i])
        return func_val