import numpy as np
from numpy import matlib as mb


class Model:

    def __init__(self,g,d,sigma,L,U,Q):
        """
        defines a generative model for diffusion of disease

        Arguments
        =========
        g : function
            describes the individual contribution of a location to the dynamics
        d : function
            defines the distance between two locations
        L : list of ntuples
            the spatial locations (in R^n) over which the model is defined
        sigma : scalar
            controls the scale of the diffusion
        U : matrix
            input matrix
        Q : matrix
            disturbance covariance matrix
        """
        # number of states
        self.n = len(L)
        # store state locations
        self.L = L
        # diffusion matrix
        F = mb.empty((self.n,self.n))
        for i,li in enumerate(L):
            for j,lj in enumerate(L):
                F[i,j] = (1./np.sqrt(2*np.pi*sigma**2))*np.exp(-d(li,lj)/(2*sigma**2))
        # normalise columns
        for j,col in enumerate(F.T):
            F[:,j] = F[:,j]/np.sum(col)
        # initialise A matrix
        self.A = mb.empty((self.n,self.n)) 
        # populate A matrix
        for i,li in enumerate(L):
            for j,lj in enumerate(L):
                self.A[i,j] = F[i,j] * g(lj)
        # store U matrix
        self.U = U
        # store disturbance covariance matrix
        self.Q = Q

    def make_C_matrix(self, r_t, alpha_t, o_t):
        c = len(o_t)
        # initialise C matrix
        C = mb.empty((c, self.n))
        # populate C matrix
        for i in range(c):
            for j in range(self.n):
                rti2 = r_t[i]**2
                const = alpha_t[i] / np.sqrt(2*np.pi*rti2)
                C[i,j] = const * np.exp(-np.abs(o_t[i]-self.L[j])**2 / 2*rti2) 
        return C

    def generate(self, x0, V, O, R, Alpha):
        """
        Arguments
        ========
        x0 : nx1 matrix
            initial state
        V : T-length list of nx1 matrices
            inputs
        O : T-length list of lists
            location of each clinic reporting at time t
        R : T-length list of lists
            radius proportional to the catchment of clinic i at time t
        Alpha : T-length list of lists
            some number proportional to the daily throughput of clinic i at time t
        """
        # check that x0 is the right size and type
        assert type(x0) is np.matrix
        assert x0.shape == (self.n,1)
        x = x0
        # assign the multivariate normal function to a local function (faster and prettier!)
        # we can do a bit of wrapping to make life easier too
        def N(mean,sigma):
            mean = np.array(mean).flatten()
            x = np.random.multivariate_normal(mean, sigma)
            return np.matrix(x).T
        # yield each new state variable by looping through the input
        for t, v in enumerate(V):
            # draw next state
            x = N(self.A*x + self.U*v, self.Q)
            # build observation matrix
            C = self.make_C_matrix(R[t], Alpha[t], O[t])
            # calculate rate parameters at each clinic
            rates = np.array(C*x).flatten()**2
            # initialise next y
            y = mb.empty((C.shape[0],1))
            for i,rate in enumerate(rates):
                # draw from the poisson
                y[i]=np.random.poisson(rate)
            yield x,y


if __name__=="__main__":
    import example.py




