import numpy as np
import pandas as pd
import numba

import matplotlib.pyplot as plt
from prettytable import PrettyTable
import progressbar
import warnings

from scipy.stats import norm
from math import factorial
from sklearn.utils import shuffle as sk_shuffle


def min_int_gt(func, thresh=0.05, x0=1, x_max=100, args=(),):
    """Find minimum integer with function evaluation greater 
    than some threshold. Function is assued to be increasing with 
    x and also that x cannot be less than 1"""
    x_lower = None # highest argument evaluated as below threshold
    x_upper = None # lowest argument evaluated as above threshold
    
    val0 = func(x0, *args)
    if val0 > thresh:
        x_lower = 0
        x_upper = x0
        x = (x_lower+x_upper)//2
    else:
        x_lower = x0
        x=x0*2
        
    min_x_found = False
    while not min_x_found:
            val = func(x, *args)
            if val > thresh:
                if x-1==x_lower:
                    min_x_found = True
                else:
                    x_upper = x
                    x = (x_lower+x_upper)//2
            else:
                x_lower = x
                if x==x_max:
                    warnings.warn("Max argument reached: returning x_max")
                    min_x_found = True
                elif x_upper==x_lower+1:
                    x = x_upper
                    min_x_found = True
                else:
                    if x_upper is None:
                        x = min(x*2, x_max)
                    else:
                        x = (x_lower+x_upper)//2      
    return x


def pmc(m,n,n_k,c):
    """In a set of n items with n_k positives and the rest negatives, calculate 
    the probability of selecting m positives when sampling c times without 
    replacement"""
    f1 = factorial(c) # 1
    f2 = factorial(n-1-c) # 2
    f3 = factorial(n_k) # 3
    f4 = factorial(n-1-n_k) # 4
    
    f5 = factorial(m)
    f6 =  factorial(c-m) # 1
    f7 =  factorial(n_k-m) # 3
    f8 =  factorial(n-1-c-n_k+m) # 2
    f9 =  factorial(n-1) # 4
    
    g1 = f1/f6
    g2 = f2/f8
    g3 = f3/f7
    g4 = f4/f9
    
    return g1*g2*g3*g4/f5


def bound_pm(m,n,n_k):
    pm=0
    for c in range(n_k, 2*n_k):
        pm += 2/n * pmc(m,n,n_k,c)
    pm += (1-2*n_k/n) * pmc(m,n,n_k,2*n_k)
    return pm


def bound_expected_consec_neigh(n,n_k):
    """
    Assuming two samples from the same distribution with n total 
    samples, how many n_k smallest distance matches would you expect
    to find within distance n_k along the sample list.
    """
    p_m = [bound_pm(m,n,n_k) for m in range(1,n_k+1)]
    mu = np.sum([p*m for p, m in zip(p_m, range(1,n_k+1))])
    var = np.sum([p*(m**2) for p, m in zip(p_m, range(1,n_k+1))]) - mu**2
    return n*mu, (var*n)**0.5


class _mst_history:
    """
    Object to store history of skips tried
    """
    def __init__(self, p_val):
        self.p_val = p_val
        self.n_skip = []
        self.n_a = []
        self.n_b = []
        self.neighbours_matched = []
        self.mu_T = []
        self.T = []
        self.sigma_T = []
        self.p_value_T = []
        self.cn = []
        self.p_value_cn = []
        
    def update(self, n_skip, n_a, n_b, neighbours_matched, mu_T, T, sigma_T,
               p_value_T, cn, p_value_cn):
        """Update stored values"""
        self.n_skip.append(n_skip)
        self.n_a.append(n_a)
        self.n_b.append(n_b)
        self.neighbours_matched.append(neighbours_matched)
        self.mu_T.append(mu_T)
        self.T.append(T)
        self.sigma_T.append(sigma_T)
        self.p_value_T.append(p_value_T)
        self.cn.append(cn)
        self.p_value_cn.append(p_value_cn)
        
    def to_dataframe(self):
        df = pd.DataFrame({
            'n_skip':self.n_skip,
            'n_a':self.n_a,
            'n_b':self.n_b,
            'neighbours_matched':self.neighbours_matched,
            'mu_T':self.mu_T,
            'T':self.T,
            'sigma_T':self.sigma_T,
            'p_value_T':self.p_value_T,
            'cn':self.cn,
            'p_value_cn':self.p_value_cn
        })
        return df
        
    def getParams(self, n_skip):
        '''
        return  (n_skip, n_a, n_b, neighbours_matched, mu_T, T, 
        p_value_T, cn, p_value_cn)
        '''
        df = self.to_dataframe()
        params = df.query('n_skip == {}'.format(n_skip)).values[0]
        return list(params)
    
    def plot(self, style='.-', yscale='linear'):
        df = self.to_dataframe()
        fig, ax = plt.subplots(1,1)
        cs = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # select optimal value
        sat_row=df.query('p_value_cn >= {}'.format(self.p_val)) \
           .sort_values('n_skip').head(1)
        # highlight the optimal value
        ax.plot(sat_row.n_skip, sat_row.p_value_cn.values, marker='x', color='k')
        ax.plot(sat_row.n_skip, sat_row.p_value_T.values, marker='x', color='k')
        # plot values searched over
        df.plot(x='n_skip', y='p_value_cn', style=style, 
                ax=ax, color=cs[0], label='Consec Neigh')
        df.plot(x='n_skip', y='p_value_T', style=style, 
                ax=ax, color=cs[1], label='T')
        # show cutoff
        ax.hlines([self.p_val], 0, df.n_skip.max()+1)
        # format
        ax.set_yscale(yscale)
        plt.legend()
        ax.set_ylabel('p-val')
        ax.set_title('p-values')
        ax.set_xlim(df.n_skip.min()-1,df.n_skip.max()+1)
        return fig


class mixed_sample_test:
    """
    Class to carry out the mixed sample PDF goodness of fit test between 
    two multivariate datasets
    """
    
    def __init__(self, w=None):
        """
        w: (1d array) inverse scaling factor used to weight distances
        """
        self.w = w
    
    def __repr__(self):
        return "mixed_sample_test object"
    
    def _fit(self, samples):
        self.w = np.nanstd(samples, axis=0)
    
    def _distance(self, sample, mixed_samples):
        """
        Calculate the distances between a single sample and an array
        of samples.
        """
        n = ((~np.isnan(sample))*(~np.isnan(mixed_samples))).sum(axis=1)
        d = np.nansum((((mixed_samples - sample)/self.w)**2), axis=1)
        d[n==0]=np.inf
        d[n!=0]=d[n!=0]/n[n!=0]
        return d
    
    def _closest_indices(self, i, samples, n_k):
        """
        Find the n_k indices which are closest to sample
        in position i.
        """
        distances = self._distance(samples[i], samples)
        indices = np.argpartition(distances, n_k+1)[:n_k+1]
        indices = indices[indices!=i]
        return indices
    
    def _calculate_mixed_sample_statistic(self, mixed_samples, classes, 
                                          n_k, log_matches, progbar=None,bar_step=100):
        n = len(mixed_samples)
        if log_matches: 
            matches = np.zeros(n*n_k)
        else:
            matches = None
        
        neighbour_same_class = 0
        consecutive_neighbour = 0
            
        # core mixed sample method
        for i in range(n):
            indices = self._closest_indices(i, mixed_samples, n_k)
            for j, ind in enumerate(indices):
                
                if classes[i] == classes[ind]:
                    neighbour_same_class+=1
                    if log_matches:
                        matches[i*n_k+j]= ind - i
                else:
                    if log_matches:
                        matches[i*n_k+j]= np.nan
                        
                if i-n_k<=ind<=i+n_k:
                    consecutive_neighbour+=1
                    
            if progbar is not None and i%bar_step==0 and i!=0:
                progbar.update(progbar.value+bar_step)
                    
        return neighbour_same_class, consecutive_neighbour, matches
    
    def _optimise_n_skip(self, n_skip, samples1, samples2, loop_n=False):
            
        n_a = len(samples1)//n_skip
        n_b = len(samples2)//n_skip
        n = n_a + n_b
        mu_T = self._calculate_mu(n_a, n_b)
        sigma_T = self._calculate_sigma(n_a, n_b, self.n_k)
        mu_cn, sigma_cn = bound_expected_consec_neigh(n, self.n_k)
        if loop_n:
            neighbours_matched_list=[]
            cn_list=[]
            T_list=[]
            p_value_T_list=[]
            p_value_cn_list=[]
            bar = progressbar.ProgressBar(max_value=n*n_skip, prefix='skip : {} |'.format(n_skip))
            for i in range(n_skip):
                # data processing
                mixed_samples = np.concatenate(
                    (samples1[i:i+n_a*n_skip:n_skip],
                     samples2[i:i+n_b*n_skip:n_skip]), axis=0)
                classes = np.concatenate(
                    (np.ones(n_a), np.zeros(n_b)), axis=0).astype(dtype=np.int32)

                # run test
                R = self._calculate_mixed_sample_statistic(mixed_samples, 
                                                      classes, self.n_k, 
                                                      log_matches=False,
                                                      progbar=bar)
                bar.update((i+1)*n)
                # collect results
                neighbours_matched = R[0]
                cn = R[1]

                # post processing
                T = (self.n_k*(n))**-1 * neighbours_matched
                p_value_T = self._calculate_p_val(T, mu_T, sigma_T)
                p_value_cn = 1-norm.cdf((cn-mu_cn)/sigma_cn)
                
                neighbours_matched_list+=[neighbours_matched]
                cn_list+=[cn]
                T_list+=[T]
                p_value_T_list+=[p_value_T]
                p_value_cn_list+=[p_value_cn]
                
            neighbours_matched = np.median(neighbours_matched_list)
            cn = np.median(cn_list)
            T = np.median(T_list)
            p_value_T = np.median(p_value_T_list)
            p_value_cn = np.median(p_value_cn_list)
                
        else:
            # data processing
            mixed_samples = np.concatenate(
                (samples1[:n_a*n_skip:n_skip], samples2[:n_b*n_skip:n_skip]), axis=0)
            classes = np.concatenate(
                (np.ones(n_a), np.zeros(n_b)), axis=0).astype(dtype=np.int32)
            
            bar = progressbar.ProgressBar(max_value=n, prefix='skip : {} |'.format(n_skip))
            # run test
            R = self._calculate_mixed_sample_statistic(mixed_samples, 
                                                  classes, self.n_k, 
                                                  log_matches=False,
                                                  progbar=bar)
            # collect results
            neighbours_matched = R[0]
            cn = R[1]

            # post processing
            T = (self.n_k*(n))**-1 * neighbours_matched
            p_value_T = self._calculate_p_val(T, mu_T, sigma_T)
            p_value_cn = 1-norm.cdf((cn-mu_cn)/sigma_cn)

        self.history.update(n_skip, n_a, n_b, neighbours_matched, 
                           mu_T, T, sigma_T, p_value_T, cn, p_value_cn)

        return p_value_cn
            
    
    def fit_gof(self, samples1, samples2, n_k, max_skip=100,
                p_val=0.05, n_skip_0=1, loop_n=False):
        """
        Fit the number point time points that must be skipped
        to decorrelate the time series and complete the gof
        test.
        """
        # instantiate variables
        self.n_k = n_k
        self._fit(np.concatenate((samples1, samples2), axis=0))
        self.shuffled = False
        self.history = _mst_history(p_val)
        self.n_skip = min_int_gt(self._optimise_n_skip, 
                        thresh=p_val, x0=n_skip_0, x_max=max_skip,
                        args=(samples1, samples2, loop_n))
        
        # retrieve variables
        (_, self.n_a, self.n_b, self.neighbours_matched, self.mu_T, 
         self.T, self.sigma_T, self.p_value, self.cn, self.p_value_cn) = \
        self.history.getParams(self.n_skip)
        self.n = self.n_a + self.n_b
            
        return self.T, self.p_value
        
    def gof(self, samples1, samples2, n_k, 
            shuffle=False, seed=None, 
            log_matches=False):
        """
        Carry out the test of goodness of fit between two samples
        """
        # data processing
        mixed_samples = np.concatenate((samples1, samples2), axis=0)
        classes = np.concatenate((
            np.ones(len(samples1)), 
            np.zeros(len(samples2))), axis=0).astype(dtype=np.int32)
        
        if shuffle:
            mixed_samples, classes = sk_shuffle(
                mixed_samples, classes, random_state=seed)
            
        # instantiate variables
        self.n_a = len(samples1)
        self.n_b = len(samples2)
        self.n_k = n_k
        self.n = self.n_a + self.n_b
        self.mu_T = self._calculate_mu(self.n_a, self.n_b)
        self.sigma_T = self._calculate_sigma(self.n_a, self.n_b, self.n_k)
        self._fit(mixed_samples)
        self.shuffled = shuffle
        
        # run test
        R = _calculate_mixed_sample_statistic(mixed_samples, classes, 
                                          n_k, log_matches)
        # collect results
        self.neighbour_same_class = R[0]
        self.consecutive_neighbour = R[1]
        if log_matches:
            self.match_log = np.unique(R[2], return_counts=True)
        else:
            self.match_log=None
                    
        # post processing
        self.T = (self.n_k*(self.n))**-1 * self.neighbour_same_class
        self.p_value = self._calculate_p_val(self.T, self.mu_T, self.sigma_T)
            
        return self.T, self.p_value
    
    @staticmethod
    def _calculate_mu(n_a, n_b):
        """Calculate the expected test statistic if samples were
        from same distribution."""
        n = n_a+n_b
        return (n_a*(n_a-1) + n_b*(n_b-1)) / (n*(n-1))
    
    @staticmethod
    def _calculate_sigma(n_a, n_b, n_k):
        """Calculate the expected standard deviation of the test 
        statistic if samples were from same distribution."""
        n = n_a+n_b
        nf = (n_a*n_b)/(n**2)
        return ((n_k*n)**-1*(nf + 4*(nf**2)))**0.5
    
    @staticmethod
    def _calculate_p_val(T, mu_T, sigma_T):
        """Calculate the associated p-value for the test statistic"""
        return 1-norm.cdf((T-mu_T)/sigma_T)
        
    def show_results(self):
        """Print table of the test results"""
        t = PrettyTable(['variable', 'value'])
        t.add_row(['p_value', self.p_value])
        t.add_row(['T', self.T])
        t.add_row(['mu_T', self.mu_T])
        t.add_row(['sigma_T', self.sigma_T])
        t.add_row(['pull', (self.T-self.mu_T)/self.sigma_T])
        t.add_row(['n_a', self.n_a])
        t.add_row(['n_b', self.n_b])
        t.add_row(['Num. consec.', self.cn])
        t.add_row(['expt. Num. consec.', '{:.2f} +- {:.2f}'.format(
            *bound_expected_consec_neigh(self.n,self.n_k))])
        t.add_row(['shuffled', self.shuffled])
        print(t)





if __name__=='__main__':
    
    from generate import gaussian, mcmc_distribution
    
    n_a = 2000
    max_skip = 30
    p_val_cn_limit = 0.05
    n_k=6
    # Create samples and apply test
    ndims=5
    p1 = gaussian(np.ones(ndims), 1.5)
    p2 = gaussian(np.ones(ndims), 1.2)
    dist1 = mcmc_distribution(p1, ndims, np.ones(ndims), 0.)
    dist2 = mcmc_distribution(p2, ndims, np.ones(ndims), 0.)
    x0 = np.ones(ndims)
    samples1 = dist1.sample(n_a, x0, burnin=100)
    samples2 = dist1.sample(n_a, x0, burnin=100)
    samples3 = dist2.sample(n_a, x0, burnin=100)
    
    mst1 = mixed_sample_test()
    mst1.fit_gof(samples1, samples2, n_k=n_k, max_skip=max_skip, p_val=p_val_cn_limit, loop_n=True)
    mst1.show_results()
    
    fig = mst1.history.plot(style='.', yscale='log')
    plt.title('MCMC same dist - averaged skip')
    plt.show()
    
    
    mst2 = mixed_sample_test()
    mst2.fit_gof(samples1, samples2, n_k=n_k, max_skip=max_skip, p_val=p_val_cn_limit, loop_n=False)
    mst2.show_results()
    fig = mst2.history.plot(style='.', yscale='log')
    plt.title('MCMC same dist - single skip')
    plt.show()

    mst3 = mixed_sample_test()
    mst3.fit_gof(samples1, samples3, n_k=n_k, max_skip=max_skip, p_val=p_val_cn_limit, loop_n=True)
    mst3.show_results()
    fig = mst3.history.plot(style='.', yscale='log')
    plt.title('MCMC different dist - averaged skip')
    plt.show()
    
    mst4 = mixed_sample_test()
    mst4.fit_gof(samples1, samples3, n_k=n_k, max_skip=max_skip, p_val=p_val_cn_limit, loop_n=False)
    mst4.show_results()
    fig = mst4.history.plot(style='.', yscale='log')
    plt.title('MCMC different dist - single skip')
    plt.show()
    


