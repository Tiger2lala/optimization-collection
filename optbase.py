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