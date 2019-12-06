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

def generate_nonmultiple_primes(maxnum, minnum=1, forbiddenmultiples=[]):
    """Generate prime numbers which are not multiples of those specified"""
    # Initialize a list
    for n in range(max(1, minnum), maxnum+1):

        # Assume number is prime until shown it is not. 
        isSoln = not any([(mult%n==0)&(n!=1) for mult in forbiddenmultiples])
        if isSoln:
            for num in range(2, int(n ** 0.5) + 1):
                if n % num == 0:
                    isSoln = False
                    break

        if isSoln:
            yield n

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
            'cn':self.consecutive_neighbour,
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
    
    
    def _calculate_consecutive_neighbours(self, sample, n_k, log_matches, 
                                          progbar=None, bar_step=100):
        n = len(sample)
        if log_matches:
            matches = np.zeros(n*n_k)
        else:
            matches = None
        
        consecutive_neighbour = 0
            
        # core mixed sample method
        for i in range(n):
            indices = self._closest_indices(i, sample, n_k)
            for j, ind in enumerate(indices):
                if log_matches:
                    matches[i*n_k+j]= ind - i
                if i-n_k<=ind<=i+n_k:
                    consecutive_neighbour+=1
                    
            if progbar is not None and i%bar_step==0 and i!=0:
                progbar.update(progbar.value+bar_step)
                    
        return consecutive_neighbour, matches
    
    
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
                    if i-n_k<=ind<=i+n_k:
                        consecutive_neighbour+=1
                else:
                    if log_matches:
                        matches[i*n_k+j]= np.nan
                    
            if progbar is not None and i%bar_step==0 and i!=0:
                progbar.update(progbar.value+bar_step)
                    
        return neighbour_same_class, consecutive_neighbour, matches
    
    def _optimise_n_skip(self, n_skip, sample, loop_n=False):
            
        n = len(sample)//n_skip
        mu_cn, sigma_cn = bound_expected_consec_neigh(n, self.n_k)
        
        if loop_n:
            cn_list=[]
            p_value_cn_list=[]
            bar = progressbar.ProgressBar(max_value=n*n_skip, prefix='skip : {} |'.format(n_skip))
            for i in range(n_skip):
                
                # data processing
                sample_i = sample[i:i+n_a*n_skip:n_skip]
                
                # run test
                cn, matches = self._calculate_consecutive_neighbours(sample_i, n_k, log_matches=False, 
                                          progbar=bar, bar_step=100)

                # post processing
                p_value_cn = 1-norm.cdf((cn-mu_cn)/sigma_cn)
                
                cn_list+=[cn]
                p_value_cn_list+=[p_value_cn]
                
            bar.update(n*n_skip)
            cn = np.median(cn_list)
            p_value_cn = np.median(p_value_cn_list)
                
        else:
            bar = progressbar.ProgressBar(max_value=n, prefix='skip : {} |'.format(n_skip))
            
            # data processing
            sample_i = sample[:n_a*n_skip:n_skip]

            # run test
            cn, matches = self._calculate_consecutive_neighbours(sample_i, n_k, log_matches=False, 
                                        progbar=None, bar_step=100)
            bar.update(n)

            # post processing
            p_value_cn = 1-norm.cdf((cn-mu_cn)/sigma_cn)

        return p_value_cn
            
    
    def fit_gof(self, sample_a, sample_b, n_k, max_skip=100,
                n_skip_0=1, p_val=0.05, loop_skip_opt=True,
                loop_test_meth='simple'):
        """
        Optimise skips and run GOF test.
        
        Fit the number point time points that must be skipped
        to decorrelate the time series and complete the gof
        test.

        Parameters
        ----------
        sample_a : (n_a, d) ndarray
        sample_b : (n_a, d) ndarray
        n_k : int
            Number of most similar points to search for in test.
        max_skip : int, optional
            Maximum number of skips allowed.
        n_skip_0 : int, optional
            Best guess to the minimum number of skips needed to 
            remove correlation.
        p_val : float, optional
            The target p-value for selecting minimum number of skips.
        loop_skip_opt : bool, optional
            Whether to perform loop in calculating minimum number of 
            skips to increase accuracy of returned values.
        loop_test_meth : {'simple', 'all2all', None}, optional
            Behaviour to apply in calculating the test statistic.
             - 'simple' - Use all uncorrelated sample subsets only once.
             - 'all2all' - Run test between all uncorrelated subsets.
             - None - Only run one subset test.
        """
        assert loop_test_meth in {'simple', 'all2all', None}, "Invalid `loop_test_calc`"
        # instantiate variables
        self.loop_skip_opt = loop_skip_opt
        self.loop_test_meth = loop_test_meth
        self.p_value_cn = p_val
        self.n_k = n_k
        self._fit(np.concatenate((samples1, samples2), axis=0))
        self.shuffled = False
        
        # optimise skip
        self.n_skip_a = min_int_gt(self._optimise_n_skip, 
                        thresh=p_val, x0=n_skip_0, x_max=max_skip,
                        args=(sample_a, loop_skip_opt))
        self.n_skip_b = min_int_gt(self._optimise_n_skip, 
                        thresh=p_val, x0=n_skip_0, x_max=max_skip,
                        args=(sample_b, loop_skip_opt))
        
        # initiate more variables
        self.n_a = len(sample_a)//self.n_skip_a
        self.n_b = len(sample_b)//self.n_skip_b
        self.n = self.n_a + self.n_b
        self.mu_T = self._calculate_mu(self.n_a, self.n_b)
        self.sigma_T = self._calculate_sigma(self.n_a, self.n_b, self.n_k)

        
        classes = np.concatenate((
            np.ones(self.n_a), 
            np.zeros(self.n_b)), axis=0).astype(dtype=np.int32)

        # loop over different uncorrelated data subsamples to get better 
        # estimate for the test statistic
        if loop_test_meth is not None:
            nbour_same_class_list = []
            consec_nbour_list = []
            T_list = []
            p_value_T_list = []
            if loop_test_meth=='simple':
                ijloop = [(i,i) for i in range(min(self.n_skip_a, self.n_skip_b))]
            elif loop_test_meth=='all2all':
                ijloop = [(i,i) for i in range(self.n_skip_a) for j in range(self.n_skip_b)]
                
            for i,j in ijloop:
                mixed_samples = np.concatenate(
                        (sample_a[i:i+self.n_a*self.n_skip_a:self.n_skip_a], 
                         sample_b[j:j+self.n_b*self.n_skip_b:self.n_skip_b]), 
                        axis=0)
                # run test
                R = self._calculate_mixed_sample_statistic(mixed_samples, classes, 
                                                  n_k, log_matches=False)
                # collect results
                nbour_same_class_list.append(R[0])
                consec_nbour_list.append(R[1])

                # post processing
                T_list.append((self.n_k*(self.n))**-1 * R[0])
                p_value_T_list.append(self._calculate_p_val(T_list[-1], self.mu_T, self.sigma_T))
                
            # collect results
            self.neighbour_same_class = np.median(nbour_same_class_list)
            self.consecutive_neighbour = np.median(consec_nbour_list)
            
            # post processing
            self.T = np.median(T_list)
            self.p_value = np.median(p_value_T_list)
            
            # store lists too
            self._T_list = T_list
            self._p_value_list = p_value_T_list
            self._neighbour_same_class_list = nbour_same_class_list
            self._consecutive_neighbour_lists = consec_nbour_list
                
        else:
            mixed_samples = np.concatenate(
                (sample_a[:self.n_a*self.n_skip_a:self.n_skip_a], 
                 sample_b[:self.n_b*self.n_skip_b:self.n_skip_b]), 
                axis=0)
            
            # run test
            R = self._calculate_mixed_sample_statistic(mixed_samples, classes, 
                                              n_k, log_matches=False)
            # collect results
            self.neighbour_same_class = R[0]
            self.consecutive_neighbour = R[1]

            # post processing
            self.T = (self.n_k*(self.n))**-1 * self.neighbour_same_class
            self.p_value = self._calculate_p_val(self.T, self.mu_T, self.sigma_T)
        
        self.match_log=None

        return self.T, self.p_value
        
    def gof(self, sample_a, sample_b, n_k, 
            shuffle=False, seed=None, 
            log_matches=False):
        """
        Carry out the test of goodness of fit between two samples
        """
        # data processing
        mixed_samples = np.concatenate((sample_a, sample_b), axis=0)
        classes = np.concatenate((
            np.ones(len(sample_a)), 
            np.zeros(len(sample_b))), axis=0).astype(dtype=np.int32)
        
        if shuffle:
            mixed_samples, classes = sk_shuffle(
                mixed_samples, classes, random_state=seed)
            
        # instantiate variables
        self.n_a = len(sample_a)
        self.n_b = len(sample_b)
        self.n_k = n_k
        self.n = self.n_a + self.n_b
        self.mu_T = self._calculate_mu(self.n_a, self.n_b)
        self.sigma_T = self._calculate_sigma(self.n_a, self.n_b, self.n_k)
        self._fit(mixed_samples)
        self.shuffled = shuffle
        
        # run test
        R = self._calculate_mixed_sample_statistic(mixed_samples, classes, 
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
        divider_row = ["-"*10, "-"*10]
        floform = lambda x: "{:.3f}".format(x)
        t = PrettyTable(['variable', 'value'])
        mu_cn, sigma_cn = bound_expected_consec_neigh(self.n,self.n_k)
        cn_pval = 1-norm.cdf((self.consecutive_neighbour-mu_cn)/sigma_cn)
        t.add_row(['p_value', floform(self.p_value)])
        t.add_row(['T', floform(self.T)])
        t.add_row(['mu_T', floform(self.mu_T)])
        t.add_row(['sigma_T', floform(self.sigma_T)])
        t.add_row(['pull', floform((self.T-self.mu_T)/self.sigma_T)])
        t.add_row(divider_row)
        t.add_row(['n_a', "{} (x{}={})".format(
            self.n_a, self.n_skip_a, self.n_a*self.n_skip_a)])
        t.add_row(['n_b', "{} (x{}={})".format(
            self.n_b, self.n_skip_b, self.n_b*self.n_skip_b)])        
        t.add_row(['n_skip_a', self.n_skip_a])
        t.add_row(['n_skip_b', self.n_skip_b])
        t.add_row(['skip p-val target', self.p_value_cn])
        t.add_row(divider_row)
        t.add_row(['Num. consec.', self.consecutive_neighbour])
        t.add_row(['Expt. Num. consec.', '{:.2f} +- {:.2f}'.format(mu_cn, sigma_cn)])
        t.add_row(['Num. Consec p-val', floform(cn_pval)])
        t.add_row(divider_row)
        t.add_row(['looped skip optimisation', self.loop_skip_opt])
        t.add_row(['decorrelated test method', self.loop_test_meth])
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
    
    # sample distributions the same, using looping
    mst1 = mixed_sample_test()    
    mst1.fit_gof(samples1, samples2, n_k, n_skip_0=1, max_skip=max_skip, p_val=p_val_cn_limit, 
                 loop_skip_opt=True, loop_test_meth='simple')
    mst1.show_results()
    
    # sample distributions the same, no looping
    mst2 = mixed_sample_test()
    mst2.fit_gof(samples1, samples2, n_k, n_skip_0=1, max_skip=max_skip, p_val=p_val_cn_limit, 
                 loop_skip_opt=False, loop_test_meth=None)    
    mst2.show_results()


    # sample distributions not the same, using looping
    mst3 = mixed_sample_test()
    mst3.fit_gof(samples1, samples3, n_k, n_skip_0=1, max_skip=max_skip, p_val=p_val_cn_limit, 
                 loop_skip_opt=True, loop_test_meth='all2all')
    mst3.show_results()
    
    # sample distributions not the same, no looping
    mst4 = mixed_sample_test()
    mst4.fit_gof(samples1, samples3, n_k, n_skip_0=1, max_skip=max_skip, p_val=p_val_cn_limit, 
                 loop_skip_opt=False, loop_test_meth=None) 
    mst4.show_results()
    


