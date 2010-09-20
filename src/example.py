
import numpy as np
from model import *

### model parameters

# little hack to choose g for each location at random
def g(l):
    choices = [0.955,0.975,0.995,1.015]
    return choices[np.random.randint(len(choices))] 
# make a little wrapper to generate normally distributed vectors
rand_vec = lambda mu,sigma: np.matrix(np.random.multivariate_normal(mu,sigma)).T

# spatial locations
n = 10
L = range(n)
# number of time points
T = 100
# distance function
d = lambda li, lj: abs(li-lj)**2
# diffusion
sigma = 1
# input matrix
U = 0.01 * np.eye(n)
# covariance matrix
Q = 0.5 * np.eye(n)
# generate random inputs
V = [rand_vec(np.zeros(n),np.eye(n)) for t in range(T)]
# pick an initial condition
x0 = mb.ones((n,1))
# clinic locations for each point in time
clinic_locations = [10*np.random.rand(np.random.randint(20)) for t in range(T)]
# radius of each clinic for each point in time (they're all the sqrt(0.2)!) 
clinic_radii = [[np.sqrt(0.2) for x in clinics] for clinics in clinic_locations]
# Alpha - not quite sure what this is
clinic_alphas = [[0.001*np.random.randint(100) for x in clinics] for clinics in clinic_locations] 

# note that in reality all this gumph generating random clinic locations/radii/alphas
# would be replaced by some database of clinics.

### create model
m = Model(g,d,sigma,L,U,Q)

### generate some data
X, Y = zip(*[xy for xy in m.generate(x0, V, clinic_locations, clinic_radii, clinic_alphas)]) 
i = 10
print len(clinic_locations[i])
print Y[i]
