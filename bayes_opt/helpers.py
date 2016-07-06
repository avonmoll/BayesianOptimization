from __future__ import print_function
from __future__ import division
import numpy as np
from datetime import datetime
from scipy.stats import norm, multivariate_normal as mvn
from scipy.linalg import cholesky, inv


class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.kappa = kappa
        
        self.xi = xi

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)

    @staticmethod
    def _ucb(x, gp, kappa):
        mean, var = gp.predict(x, eval_MSE=True)
        return mean + kappa * np.sqrt(var)

    @staticmethod
    def _ei(x, gp, y_max, xi):
        mean, var = gp.predict(x, eval_MSE=True)

        # Avoid points with zero variance
        var = np.maximum(var, 1e-9 + 0 * var)

        z = (mean - y_max - xi) / np.sqrt(var)
        return (mean - y_max - xi) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        mean, var = gp.predict(x, eval_MSE=True)

        # Avoid points with zero variance
        var = np.maximum(var, 1e-9 + 0 * var)

        z = (mean - y_max - xi) / np.sqrt(var)
        return norm.cdf(z)


def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]
    

def slice_sample(init_x, logprob, sigma=1.0, step_out=True, max_steps_out=1000,
                 compwise=False):
    """
    Slice sampling for GP hyperparameters. Used when integrateOverHypers is
    True. Based off of https://github.com/JasperSnoek/spearmint/blob/master/spearmint/spearmint/util.py#L34.
    
    :param init_x: array, initial value(s) or previous sample
    
    :param logprob: callable, function to evaluate the log probability of the
    sample - used as proposal evaluation criteria
    
    :param sigma: float, width of proposal range
    
    :param step_out: Bool, whether to grow the proposal range
    
    :param max_steps_out: int, maximum iterations for expanding the proposal
    range
    
    :param compwise: Bool, whether each dimension should be sampled individually
    
    :return: array of the same size as x, new sample
    """
    
    def direction_slice(direction, init_x):
        def dir_logprob(z):
            return logprob(direction * z + init_x)
    
        upper = sigma * np.random.rand()
        lower = upper - sigma
        llh_s = np.log(np.random.rand()) + dir_logprob(0.0)
    
        l_steps_out = 0
        u_steps_out = 0
        if step_out:
            while dir_logprob(lower) > llh_s and l_steps_out < max_steps_out:
                l_steps_out += 1
                lower -= sigma
            while dir_logprob(upper) > llh_s and u_steps_out < max_steps_out:
                u_steps_out += 1
                upper += sigma
            
        steps_in = 0
        while True:
            steps_in += 1
            new_z = (upper - lower) * np.random.rand() + lower
            new_llh = dir_logprob(new_z)
            if np.isnan(new_llh):
                print(new_z, direction * new_z + init_x, new_llh, llh_s,
                      init_x, logprob(init_x))
                raise Exception("Slice sampler got a NaN")
            if new_llh > llh_s:
                break
            elif new_z < 0:
                lower = new_z
            elif new_z > 0:
                upper = new_z
            else:
                raise Exception("Slice sampler shrank to zero!")

        return new_z * direction + init_x
    
    if not init_x.shape:
        init_x = np.array([init_x])

    dims = init_x.shape[0]
    if compwise:
        ordering = range(dims)
        np.random.shuffle(ordering)
        cur_x = init_x.copy()
        for d in ordering:
            direction = np.zeros((dims))
            direction[d] = 1.0
            cur_x = direction_slice(direction, cur_x)
        return cur_x
            
    else:
        direction = np.random.randn(dims)
        direction = direction / np.sqrt(np.sum(direction**2))
        return direction_slice(direction, init_x)
        
def elliptical_slice(initial_theta,prior,lnpdf,
                     cur_lnpdf=None,angle_range=None):
    """
    NAME:
       elliptical_slice
    PURPOSE:
       Markov chain update for a distribution with a Gaussian "prior" factored out
    INPUT:
       initial_theta - initial vector
       prior - cholesky decomposition of the covariance matrix 
               (like what numpy.linalg.cholesky returns), 
               or a sample from the prior
       lnpdf - function evaluating the log of the pdf to be sampled
       pdf_params= parameters to pass to the pdf
       cur_lnpdf= value of lnpdf at initial_theta (optional)
       angle_range= Default 0: explore whole ellipse with break point at
                    first rejection. Set in (0,2*pi] to explore a bracket of
                    the specified width centred uniformly at random.
    OUTPUT:
       new_theta, new_lnpdf
    HISTORY:
       Originally written in matlab by Iain Murray (http://homepages.inf.ed.ac.uk/imurray2/pub/10ess/elliptical_slice.m)
       2012-02-24 - Written - Bovy (IAS)
       2016-06-29 - Adapted for multidimensional theta - Von Moll
    """
    D= len(initial_theta)
    theta = initial_theta
    
    order = range(len(theta))
    np.random.shuffle(order)
    for i in order:
        if cur_lnpdf is None:
            cur_lnpdf= lnpdf(theta)

        # Set up the ellipse and the slice threshold
        nu = prior(i)

        hh = np.log(np.random.uniform()) + cur_lnpdf

        # Set up a bracket of angles and pick a first proposal.
        # "phi = (theta'-theta)" is a change in angle.
        if angle_range is None or angle_range == 0.:
            # Bracket whole ellipse with both edges at first proposed point
            phi= np.random.uniform()*2.*np.pi
            phi_min= phi-2.*np.pi
            phi_max= phi
        else:
            # Randomly center bracket on current point
            phi_min= -angle_range*np.random.uniform()
            phi_max= phi_min + angle_range
            phi= np.random.uniform()*(phi_max-phi_min)+phi_min

        # Slice sampling loop
        theta_p = np.copy(theta)
        while True:
            # Compute xx for proposed angle difference and check if it's on the slice
            xx_prop = theta[i]*np.cos(phi) + nu*np.sin(phi)
            theta_p[i] = xx_prop
            cur_lnpdf = lnpdf(theta_p)
            if cur_lnpdf > hh:
                # New point is on slice, ** EXIT LOOP **
                theta[i] = xx_prop
                break
            # Shrink slice to rejected point
            if phi > 0:
                phi_max = phi
            elif phi < 0:
                phi_min = phi
            else:
                raise RuntimeError('BUG DETECTED: Shrunk to current position and still not acceptable.')
            # Propose new angle difference
            phi = np.random.uniform()*(phi_max - phi_min) + phi_min
    return theta, cur_lnpdf

# TODO: consider adding helper function to construct R matrix to avoid calling
#       gp.fit() too often. This will be useful for the hyperparameter slice
#       sampling process. Could be more memory efficient than having a bunch of
#       GP objects, but then again having separate objects may be easier.
#       Although, for burnin purposes, we would want to definitely only have a
#       single GP.

# TODO: Write a function to predict a single point from the GP a la Rasmussen
#       using noise and amplitude -- OR -- could try to fiddle with using
#       scikit-learn's GP by scaling the correlation function and adding a
#       so-called "nugget" for the noise


class BColours(object):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


class PrintLog(object):

    def __init__(self, params):

        self.ymax = None
        self.xmax = None
        self.params = params
        self.ite = 1

        self.start_time = datetime.now()
        self.last_round = datetime.now()

        # sizes of parameters name and all
        self.sizes = [max(len(ps), 7) for ps in params]

        # Sorted indexes to access parameters
        self.sorti = sorted(range(len(self.params)),
                            key=self.params.__getitem__)

    def reset_timer(self):
        self.start_time = datetime.now()
        self.last_round = datetime.now()

    def print_header(self, initialization=True):

        if initialization:
            print("{}Initialization{}".format(BColours.RED,
                                              BColours.ENDC))
        else:
            print("{}Bayesian Optimization{}".format(BColours.RED,
                                                     BColours.ENDC))

        print(BColours.BLUE + "-" * (29 + sum([s + 5 for s in self.sizes])) + BColours.ENDC)

        print("{0:>{1}}".format("Step", 5), end=" | ")
        print("{0:>{1}}".format("Time", 6), end=" | ")
        print("{0:>{1}}".format("Value", 10), end=" | ")

        for index in self.sorti:
            print("{0:>{1}}".format(self.params[index],
                                    self.sizes[index] + 2),
                  end=" | ")
        print('')

    def print_step(self, x, y, warning=False):

        print("{:>5d}".format(self.ite), end=" | ")

        m, s = divmod((datetime.now() - self.last_round).total_seconds(), 60)
        print("{:>02d}m{:>02d}s".format(int(m), int(s)), end=" | ")

        if self.ymax is None or self.ymax < y:
            self.ymax = y
            self.xmax = x
            print("{0}{2: >10.5f}{1}".format(BColours.MAGENTA,
                                             BColours.ENDC,
                                             y),
                  end=" | ")

            for index in self.sorti:
                print("{0}{2: >{3}.{4}f}{1}".format(BColours.GREEN, BColours.ENDC,
                                                    x[index],
                                                    self.sizes[index] + 2,
                                                    min(self.sizes[index] - 3, 6 - 2)),
                      end=" | ")
        else:
            print("{: >10.5f}".format(y), end=" | ")
            for index in self.sorti:
                print("{0: >{1}.{2}f}".format(x[index],
                                              self.sizes[index] + 2,
                                              min(self.sizes[index] - 3, 6 - 2)),
                      end=" | ")

        if warning:
            print("{}Warning: Test point chose at "
                  "random due to repeated sample.{}".format(BColours.RED,
                                                            BColours.ENDC))

        print()

        self.last_round = datetime.now()
        self.ite += 1

    def print_summary(self):
        pass
