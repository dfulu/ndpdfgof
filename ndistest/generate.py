import numpy as np
import numba

from functools import reduce


def matmul(*args):
    """
    Performs matrix multiplication on the array arguments input in order
    """
    return reduce(np.matmul,  args)

def mcmc_factory(func, N, x0, dims, sigma, burnin=0):
    """Function to return a numba compatible MCMC sampling function 
    for any generative function compatable with Numba
    
    Note still under construction
    """
    
    def numba_func(x):
        return func(x)
    
    
    def mcmc_sample(N, x0, dims, sigma, burnin):
        samples=np.zeros((N, dims), dtype=np.float32)
        f0 = numba_func(x0)
        for i in range(burnin):
            accept=False
            while not accept:
                x_ = np.random.normal(x0,sigma)
                f_ = numba_func(x_)
                alpha = f_/f0
                if alpha>np.random.uniform():
                    accept=True
                    x0=x_
                    f0=f_

        samples[0] = x0
        for i in range(1,N):
            accept=False
            while not accept:
                x_ = np.random.normal(x0,sigma)
                f_ = numba_func(x_)
                alpha = f_/f0
                if alpha>np.random.uniform():
                    accept=True
                    x0=x_
                    f0=f_
                    samples[i]=x0
        return samples
    
    return mcmc_sample(N, x0, dims, sigma, burnin)


class mcmc_distribution:
    """
    MCMC sample generation class
    """
    
    def __init__(self, func, dims, sigma, missfrac=0, seed=None):
        '''
        Args:
            func: (numba-ready function) unnormalised probability function
            dims: (int) number of dimensions
            missfrac: (float) the fraction of values that should be missing
            seed: (int) random seed
            sigma: (1D array) width of gaussian used to sample step sizes
        '''
        self.func = func
        self.dims=dims
        self.sigma = sigma
        self.missfrac = missfrac
        np.random.seed(seed)
    
    def sample(self, N, x0, burnin=0):
        """
        N: (int) numbr of samples
        x0: (1d array) starting point
        burnin: (int) number of samples to burn in for
        """
        samples = mcmc_factory(self.func, N, x0, self.dims, self.sigma, burnin=0)
        
        if self.missfrac!=0:
            missing = np.ones(shape=samples.shape, dtype=bool)
            trivial = np.ones(shape=(N,), dtype=bool)
            n_trivial = N
            # generate mask where not all datapoints are masked in a row
            while n_trivial!=0:
                missing_ = np.random.choice([True, False], 
                                            size=(n_trivial, self.dims), 
                                            replace=True, 
                                            p=[self.missfrac, 1-self.missfrac])
                missing[trivial]=missing_
                trivial = missing.sum(axis=1)==self.dims
                n_trivial = trivial.sum()
            samples[missing]=np.nan
        return samples
    

def gaussian(xc, sigma):
    """
    Gaussian functional for isotropic multivariate guassian
    xc: (1d array) centre value
    sigma: (float) full width half max
    """
    def f(x):
        return np.exp(-0.5*(((x-xc)/sigma)**2).sum())/((2*np.pi)**0.5*sigma)
    return f


def multivariate_gaussian(mu, sigma):
    """
    Multivariate Gaussian functional for multivariate guassian
    mu: (1d array) centre value
    sigma: (2d array) covraince matrix
    """
    sigma_inv = np.linalg.inv(sigma)
    A = (np.linalg.det(sigma)*(2*np.pi)**len(mu))**-0.5
    def f(x):
        return A*np.exp(-0.5*matmul((x-mu).T, sigma_inv, (x-mu)))
    return f


if __name__=='__main__':
    
    n_a = 5000
    # Create samples and apply test
    ndims=20
    p1 = gaussian(np.ones(ndims), 1.5)
    p2 = gaussian(np.ones(ndims), 1.2)
    dist1 = mcmc_distribution(p1, ndims, np.ones(ndims), 0.)
    dist2 = mcmc_distribution(p2, ndims, np.ones(ndims), 0.)
    x0 = np.ones(ndims)
    samples1 = dist1.sample(n_a, x0, burnin=100)
    samples2 = dist1.sample(n_a, x0, burnin=100)
    samples3 = dist2.sample(n_a, x0, burnin=100)
    
    import matplotlib.pyplot as plt
    plt.scatter(samples1[:,0], samples1[:,1], alpha=0.2)
    plt.scatter(samples3[:,0], samples3[:,1], alpha=0.1)
    plt.show()
    