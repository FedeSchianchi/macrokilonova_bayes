import numpy as np
import math

# define density of probability function for chi2
def density_of_probability1(x, dof):
    d = np.exp( -dof/2 * np.log(2) + (dof/2 - 1) * np.log(x) -x/2 ) / ( math.gamma( dof/2 ) )
    return d