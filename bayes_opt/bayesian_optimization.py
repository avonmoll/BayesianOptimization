from __future__ import print_function
from __future__ import division
import numpy as np
from sklearn.gaussian_process import GaussianProcess
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve
from .helpers import UtilityFunction, unique_rows, PrintLog, slice_sample

__author__ = 'fmfn'


def acq_max(ac, gp, y_max, bounds):
    """
    A function to find the maximum of the acquisition function using
    the 'L-BFGS-B' method.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.


    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """

    def integrateAcqOverHypers(x):
        acq_list = np.zeros((len(gp_list),))
        for i in xrange(len(gp_list)):
            acq_list[i] = ac(x.reshape(1, -1), gp=gp_list[i], y_max=y_max)
        
        return np.mean(acq_list)
    
    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    max_acq = None

    x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                size=(100, bounds.shape[0]))

    for x_try in x_tries:
        if type(gp) == list:  # integrateOverHypers is True
            gp_list = gp
            res = minimize(lambda x: -integrateAcqOverHypers(x),
                           x_try.reshape(1, -1),
                           bounds=bounds,
                           method="L-BFGS-B")
        else:
            # Find the minimum of minus the acquisition function
            res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                           x_try.reshape(1, -1),
                           bounds=bounds,
                           method="L-BFGS-B")

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -res.fun >= max_acq:
            x_max = res.x
            max_acq = -res.fun

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])


def matern52(theta, d):
    """
    Matern 5/2 correlation model.::
    
        theta, d --> r(theta, d) = (1+sqrt(5)*r + 5/3*r^2)*exp(-sqrt(5)*r)
        
                               n
            where r = sqrt(   sum  (d_i)^2 / (theta_i)^2 )
                             i = 1
                             
    Parameters
    ----------
    theta : array_like
        An array with shape 1 (isotropic) or n (anisotropic) giving the
        autocorrelation parameter(s).
        
    d : array_like
        An array with shape (n_eval, n_features) giving the componentwise
        distances between locations x and x' at which the correlation model
        should be evaluated.
        
    Returns
    -------
    r : array_like
        An array with shape (n_eval, ) containing the values of the
        autocorrelation model.
    """

    theta = np.asarray(theta, dtype=np.float)
    d = np.asarray(d, dtype=np.float)
    
    if d.ndim > 1:
        n_features = d.shape[1]
    else:
        n_features = 1
    
    if theta.size == 1:
        r = np.sqrt(np.sum(d ** 2, axis=1)) / theta[0]
    elif theta.size != n_features:
        raise ValueError("Length of theta must be 1 or %s" % n_features)
    else:
        r = np.sqrt(np.sum(d ** 2 / theta.reshape(1, n_features) ** 2, axis=1))

    return (1 + np.sqrt(5) * r + 5 / 3. * r ** 2) * np.exp(-np.sqrt(5) * r)
    

class BayesianOptimization(object):

    def __init__(self, f, pbounds, verbose=1):
        """
        :param f:
            Function to be maximized.

        :param pbounds:
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        :param verbose:
            Whether or not to print progress.

        """
        # Store the original dictionary
        self.pbounds = pbounds

        # Get the name of the parameters
        self.keys = list(pbounds.keys())

        # Find number of parameters
        self.dim = len(pbounds)

        # Create an array with parameters bounds
        self.bounds = []
        for key in self.pbounds.keys():
            self.bounds.append(self.pbounds[key])
        self.bounds = np.asarray(self.bounds)

        # Some function to be optimized
        self.f = f

        # Initialization flag
        self.initialized = False

        # Initialization lists --- stores starting points before process begins
        self.init_points = []
        self.x_init = []
        self.y_init = []

        # Numpy array place holders
        self.X = None
        self.Y = None
        
        # Hyperparameter place holders
        self.mean = None
        self.amp2 = None
        self.noise = None
        self.ls = None
        self.hyper_samples = []
        self.noiseless = None
        
        # Hyperparameter Scales
        self.noise_scale = 0.1
        self.amp2_scale = 1.0
        self.max_ls = 2.0

        # Counter of iterations
        self.i = 0

        # Since scipy 0.16 passing lower and upper bound to theta seems to be
        # broken. However, there is a lot of development going on around GP
        # is scikit-learn. So I'll pick the easy route here and simple specify
        # only theta0.
        self.gp = GaussianProcess(corr=matern52,
                                  theta0=np.random.uniform(0.001, 0.05,
                                                           self.dim),
                                  thetaL=1e-5 * np.ones(self.dim),
                                  thetaU=1e0 * np.ones(self.dim),
                                  random_start=30)
        
        # Placeholder for list of GPs used when integrateOverHypers is True
        self.gp_list = []

        # Utility Function placeholder
        self.util = None

        # PrintLog object
        self.plog = PrintLog(self.keys)

        # Output dictionary
        self.res = {}
        # Output dictionary
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values': [], 'params': []}

        # Verbose
        self.verbose = verbose

    def init(self, init_points):
        """
        Initialization method to kick start the optimization process. It is a
        combination of points passed by the user, and randomly sampled ones.

        :param init_points:
            Number of random points to probe.
        """

        # Generate random points
        l = [np.random.uniform(x[0], x[1], size=init_points) for x in self.bounds]

        # Concatenate new random points to possible existing
        # points from self.explore method.
        self.init_points += list(map(list, zip(*l)))

        # Create empty list to store the new values of the function
        y_init = []

        # Evaluate target function at all initialization
        # points (random + explore)
        for x in self.init_points:

            y_init.append(self.f(**dict(zip(self.keys, x))))

            if self.verbose:
                self.plog.print_step(x, y_init[-1])

        # Append any other points passed by the self.initialize method (these
        # also have a corresponding target value passed by the user).
        self.init_points += self.x_init

        # Append the target value of self.initialize method.
        y_init += self.y_init

        # Turn it into np array and store.
        self.X = np.asarray(self.init_points)
        self.Y = np.asarray(y_init)

        # Updates the flag
        self.initialized = True

    def explore(self, points_dict):
        """
        Method to explore user defined points

        :param points_dict:
        :return:
        """

        # Consistency check
        param_tup_lens = []

        for key in self.keys:
            param_tup_lens.append(len(list(points_dict[key])))

        if all([e == param_tup_lens[0] for e in param_tup_lens]):
            pass
        else:
            raise ValueError('The same number of initialization points '
                             'must be entered for every parameter.')

        # Turn into list of lists
        all_points = []
        for key in self.keys:
            all_points.append(points_dict[key])

        # Take transpose of list
        self.init_points = list(map(list, zip(*all_points)))

    def initialize(self, points_dict):
        """
        Method to introduce point for which the target function
        value is known

        :param points_dict:
        :return:
        """

        for target in points_dict:

            self.y_init.append(target)

            all_points = []
            for key in self.keys:
                all_points.append(points_dict[target][key])

            self.x_init.append(all_points)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        :param new_bounds:
            A dictionary with the parameter name and its new bounds

        """

        # Update the internal object stored dict
        self.pbounds.update(new_bounds)

        # Loop through the all bounds and reset the min-max bound matrix
        for row, key in enumerate(self.pbounds.keys()):

            # Reset all entries, even if the same.
            self.bounds[row] = self.pbounds[key]
            
    def _generate_hypers_samples(self):
        """ Largely based off of https://github.com/JasperSnoek/spearmint/blob/master/spearmint/spearmint/chooser/GPEIOptChooser.py#L621
        """
        
        def sample_hypers(burningIn=False):
            sample_mean_noise_amp2()
            if self.noiseless:
                self.noise = 1e-3
            sample_ls()
            if not burningIn:
                self.hyper_samples.append(
                    (self.mean, self.noise, self.amp2, self.ls))
            
        def sample_mean_noise_amp2():
            def logprob(hypers):
                mean = hypers[0]
                amp2 = hypers[1]
                noise = hypers[2]
                
                if mean > np.max(self.Y) or mean < np.min(self.Y):
                    return -np.inf
                
                if amp2 < 0 or noise < 0:
                    return -np.inf
                
                R = np.dot(self.gp.C, self.gp.C.transpose())
                
                cov = (amp2 * (R + 1e-6 * np.eye(R.shape[0])) +
                       noise * np.eye(R.shape[0]))
                chol = cholesky(cov, lower=True)
                solve = cho_solve((chol, True),
                                  (self.Y - mean))
                lp = -np.sum(np.log(np.diag(chol))) \
                     - 0.5 * np.dot((self.Y - mean), solve)
                
                if not self.noiseless:
                    # Add noise horseshoe prior
                    lp += np.log(np.log(1 + (self.noise_scale / noise)**2))
                    
                # Add in amplitude lognormal prior
                lp -= 0.5 * (np.log(np.sqrt(amp2)) / self.amp2_scale)**2
                
                return lp
            
            hypers = slice_sample(np.array(
                [self.mean, self.amp2, self.noise]), logprob, compwise=False)
            self.mean = hypers[0]
            self.amp2 = hypers[1]
            self.noise = hypers[2]
            
        def sample_ls():
            def logprob(ls):
                if np.any(ls < 0) or np.any(ls > self.max_ls):
                    return -np.inf
                
                # Update GP
                self.gp.set_params(**{'theta0': ls,
                                      'thetaU': None,
                                      'thetaL': None})
                ur = unique_rows(self.X)
                self.gp.fit(self.X[ur], self.Y[ur])
                
                R = np.dot(self.gp.C, self.gp.C.transpose())
                
                cov = (self.amp2 * (R + 1e-6 * np.eye(R.shape[0])) +
                       self.noise * np.eye(R.shape[0]))
                chol = cholesky(cov, lower=True)
                solve = cho_solve((chol, True),
                                  (self.Y - self.mean))
                lp = -np.sum(np.log(np.diag(chol))) \
                     - 0.5 * np.dot((self.Y - self.mean),
                     solve)
                return lp
            
            self.ls = slice_sample(self.ls, logprob, compwise=True)
            
        # Initialize hyperparameters
        self.mean = np.mean(self.Y)
        self.noise = 1e-3
        self.amp2 = np.std(self.Y) + 1e-4
        self.ls = np.ones((self.dim,))
        
        # Perform burn-in
        for i in xrange(100):
            sample_hypers(burningIn=True)
        
        # Clear previous hyper samples
        self.hyper_samples = []
        
        # Store hyper parameter samples
        for i in xrange(10):
            sample_hypers()
            
    def _construct_gp_list(self):
        # Construct list of GPs based on hyper_samples
        if not self.gp_list:
            self.gp_list = [GaussianProcess() for hyper in
                            xrange(len(self.hyper_samples))]
        ur = unique_rows(self.X)
        for i, (mean, noise, amp2, ls) in enumerate(self.hyper_samples):
            gp_params = self.gp.get_params()
            gp_params['theta0'] = ls
            if not self.noiseless:
                gp_params['nugget'] = noise
            # TODO: Perhaps we need to create our own version of GP prediction
            #       so that we can utilize the mean and amp2 samples properly.
            #       I tried simply injecting them into the scikit-learn GPs,
            #       but it did not lead to good results.
            self.gp_list[i].set_params(**gp_params)
            self.gp_list[i].fit(self.X[ur], self.Y[ur])
            # self.gp_list[i].y_mean = mean
            # self.gp_list[i].y_std = np.sqrt(amp2)

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ei',
                 kappa=2.576,
                 xi=0.0,
                 integrateOverHypers=False,
                 noiseless=True,
                 **gp_params):
        """
        Main optimization method.

        Parameters
        ----------
        :param init_points:
            Number of randomly chosen points to sample the
            target function before fitting the gp.

        :param n_iter:
            Total number of times the process is to repeated. Note that
            currently this methods does not have stopping criteria (due to a
            number of reasons), therefore the total number of points to be
            sampled must be specified.

        :param acq:
            Acquisition function to be used, defaults to Expected Improvement.
            
        :param kappa:
            Scalar float for controlling tradeoff between exporation and
            exploitation when using the GP-UCB acquisition function
            
        :param xi:
            Scalar float for controlling tradeoff between exploration and
            exploitatoin when using EI or POI acquisition functions

        :param gp_params:
            Parameters to be passed to the Scikit-learn Gaussian Process object

        :param integrateOverHypers:
            Boolean to decide whether or not to sample from a posterior
            hyperparameter distribution and integrate over the possibilities or
            to estimate the hyperparameters (lengthscales) using maximum
            likelihood (default)
            
        :param noiseless:
            Boolean to specify whether the observations are noisy or not (only
            used when integrateOverHypers is True)
        
        Returns
        -------
        :return: Nothing
        """
        # Reset timer
        self.plog.reset_timer()

        # Set acquisition function
        self.util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)
        
        self.noiseless = noiseless

        # Initialize x, y and find current y_max
        if not self.initialized:
            if self.verbose:
                self.plog.print_header()
            self.init(init_points)

        y_max = self.Y.max()

        # Set parameters if any was passed
        if integrateOverHypers:
            gp_params['theta0'] = np.ones((self.dim,))
            gp_params['thetaU'] = None
            gp_params['thetaL'] = None
        self.gp.set_params(**gp_params)

        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        if integrateOverHypers:
            self._generate_hypers_samples()
            self._construct_gp_list()
        
        # Finding argmax of the acquisition function.
        x_max = acq_max(ac=self.util.utility,
                        gp=self.gp_list if integrateOverHypers else self.gp,
                        y_max=y_max,
                        bounds=self.bounds)

        # Print new header
        if self.verbose:
            self.plog.print_header(initialization=False)
        # Iterative process of searching for the maximum. At each round the
        # most recent x and y values probed are added to the X and Y arrays
        # used to train the Gaussian Process. Next the maximum known value
        # of the target function is found and passed to the acq_max function.
        # The arg_max of the acquisition function is found and this will be
        # the next probed value of the target function in the next round.
        for i in range(n_iter):
            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            pwarning = False
            if np.any((self.X - x_max).sum(axis=1) == 0):

                x_max = np.random.uniform(self.bounds[:, 0],
                                          self.bounds[:, 1],
                                          size=self.bounds.shape[0])

                pwarning = True

            # Append most recently generated values to X and Y arrays
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))
            self.Y = np.append(self.Y, self.f(**dict(zip(self.keys, x_max))))

            # Updating the GP.
            ur = unique_rows(self.X)
            
            if integrateOverHypers:
                gp_params = self.gp.get_params()
                gp_params['theta0'] = np.ones((self.dim,))
                gp_params['thetaU'] = None
                gp_params['thetaL'] = None
                self.gp.set_params(**gp_params)
                
            self.gp.fit(self.X[ur], self.Y[ur])
            
            if integrateOverHypers:
                self._generate_hypers_samples()
                self._construct_gp_list()

            # Update maximum value to search for next probe point.
            if self.Y[-1] > y_max:
                y_max = self.Y[-1]

            # Maximize acquisition function to find next probing point
            x_max = acq_max(ac=self.util.utility,
                            gp=self.gp_list if integrateOverHypers else self.gp,
                            y_max=y_max,
                            bounds=self.bounds)

            # Print stuff
            if self.verbose:
                self.plog.print_step(self.X[-1], self.Y[-1], warning=pwarning)

            # Keep track of total number of iterations
            self.i += 1

            self.res['max'] = {'max_val': self.Y.max(),
                               'max_params': dict(zip(self.keys,
                                                      self.X[self.Y.argmax()]))
                               }
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(dict(zip(self.keys, self.X[-1])))

        # Print a final report if verbose active.
        if self.verbose:
            self.plog.print_summary()
