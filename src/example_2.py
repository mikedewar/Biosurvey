"""
This script runs a 2D example of a LDS with Poisson observations.
"""

import pylab as pb
import numpy as np
from model import *

# make a little wrapper to generate normally distributed vectors
rand_vec = lambda mu,sigma: np.matrix(np.random.multivariate_normal(mu,sigma)).T

### model parameters
# spatial locations
m = 10
L = []
for i in range(m):
    for j in range(m):
        L.append((i,j))


n = len(L)
# number of time points
T = 100
# distance function
d = lambda li, lj: np.sqrt((li[0]-lj[0])**2 + (li[1]-lj[1])**2)**2
# little hack to choose g for each location at random
def g(l):
    choices = [0.955,0.975,0.995,1.015]
    return choices[np.random.randint(len(choices))]



# diffusion
sigma = 1
# input matrix
U = 0.01 * np.eye(n)
# covariance matrix
K = lambda li,lj: np.exp(-d(li,lj)/10.)
Q = np.empty((n,n))
for i,li in enumerate(L):
    for j,lj in enumerate(L):
        Q[i,j]=K(li,lj)



# generate random inputs
V = [rand_vec(np.zeros(n),np.eye(n)) for t in range(T)]
# pick an initial condition
x0 = mb.ones((n,1))
# clinic locations for each point in time
def random_locns(max_num_locns): 
    return zip(
        (m-1)*np.random.rand(np.random.randint(max_num_locns)),
        (m-1)*np.random.rand(np.random.randint(max_num_locns))
    )

clinic_locations = [random_locns(20) for t in range(T)]
# radius of each clinic for each point in time (they're all the same!) 
clinic_radii = [[2. for x in clinics] for clinics in clinic_locations]
# Alpha - not quite sure what this is
#clinic_alphas = [[0.001*np.random.randint(100) for x in clinics] for clinics in clinic_locations] 
# let's pretend all the clinics get 100 people in a day
avg_througput = 1000
clinic_alphas = [
    [
        0.001*avg_througput for x in clinics
    ] for clinics in clinic_locations
] 

# note that in reality all this gumph generating random clinic locations/radii/alphas
# would be replaced by some database of clinics.

### create model
kam = Model(g,d,sigma,L,U,Q)

### generate some data
data_maker = kam.generate(x0, V, clinic_locations, clinic_radii, clinic_alphas)
X, Y = zip(*[xy for xy in data_maker]) 

# make a pic!
ts = [10,20,30,40,50,60]
for t in ts:
    pb.figure()
    Z = pb.array(abs(X[t]).reshape(m,m))
    pb.matshow(Z)
    pb.colorbar()
    xy = clinic_locations[t]
    x1,x2 = zip(*clinic_locations[t])
    pb.scatter(x1,x2,s=np.array(Y[t]).flatten())
    pb.xlim((0,m-1))
    pb.ylim((0,m-1))
pb.show()


