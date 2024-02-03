# Test functions for optimizers
# @tz2lala

import numpy as np

def gen_linear_test(m: int = 1000) -> np.ndarray:
    diag = 0.5 + np.sqrt(np.arange(1, m+1))
    ret_mat = np.diag(np.ones(m-100), -100) + \
              np.diag(np.ones(m-1), -1) + \
              np.diag(diag) + \
              np.diag(np.ones(m-1), 1) + \
              np.diag(np.ones(m-100), 100)
    return ret_mat

EGG_HOLDER = {'obj': lambda x: (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))
                     -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47))))),
              'bounds': [(-512, 512), (-512, 512)]}

MY_QUARTIC = {'obj': lambda x: x**4 + x**3 - 7*x**2 - x + 6,
              'bounds': [(-3, 3)]}

ROSENBROCK = {'obj': lambda x: 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2,
              'grad': lambda x: np.array([-400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
                                            200*(x[1] - x[0]**2)]),
              'bounds': [(-3, 3), (-3, 3)]}

MY_LINEAR = {'A': gen_linear_test(1000),
             'b': np.ones(1000)}


