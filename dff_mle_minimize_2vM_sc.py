#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 19:10:48 2018
Author Dimitrios Fafalis 

This code generates a random sample from a mixture of TWO von Mises 
    on the SEMI-CIRCLE and ONE uniform distribution, 
    with weights p1, p2 and (1-p1-p2) 
    using the functions: "stats.vonmises.rvs" and "stats.uniform.rvs"
    using the "numpy.random.choice" function
    
This code makes use of "optimize.minimize" to minimize: 
    a. the negative log-likelihood function of the mixture of TWO von Mises
        distributions, with constraints 

@author: df
"""

import numpy as np
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------- #
# some initial parameters to generate the random sample: 
# size of sample: 
N = 1000
# parameters for the von Mises members:
p1_ = 0.4 # weight contribution of the 1st von Mises 
p2_ = 0.6 # weight contribution of the 1st von Mises 
#pu = 1. - p1_ - p2_ # weight contribution of the uniform 
kappa1_ = np.array((12.0)) # concentration for the 1st von Mises member 
kappa2_ = np.array((12.0)) # concentration for the 1st von Mises member 
loc1_ = -np.pi/6.0 # location for the 1st von Mises member 
loc2_ =  np.pi/6.0 # location for the 1st von Mises member 
# collective arrays of the concentrations, locations and weights: 
kappas = np.array([kappa1_, kappa2_]) 
locs = np.array([loc1_, loc2_])
#pis = np.array([p1_, p2_, pu])
# parameter to scale the user-defined von Mises on the SEMI-CIRCLE 
ll = 2.0
# parameter to scale the stats von Mises on the SEMI-CIRCLE 
scal_ = 0.5
# ---------------------------------------------------------------------- #

# ---------------------------------------------------------------------- #
# Generate the random sample:
# "Creating a mixture of probability distributions for sampling"
#   a question on stackoverflow: 
#   https://stackoverflow.com/questions/47759577/
#               creating-a-mixture-of-probability-distributions-for-sampling
u1_ = -np.pi/2
u2_ =  np.pi
distributions = [
        { "type": stats.vonmises.rvs, \
                     "args": {"kappa":kappa1_, "loc":loc1_, "scale":scal_ }},
        { "type": stats.vonmises.rvs, \
                     "args": {"kappa":kappa2_, "loc":loc2_, "scale":scal_ }},
#        { "type": stats.uniform.rvs,  \
#                     "args": {"loc":u1_, "scale":u2_ } }
]
#coefficients = np.array( [ p1_, p2_, pu ] ) # these are the weights 
coefficients = np.array( [ p1_, p2_ ] ) # these are the weights 
coefficients /= coefficients.sum() # in case these did not add up to 1
sample_size = N

num_distr = len(distributions)
data = np.zeros((sample_size, num_distr))
for idx, distr in enumerate(distributions):
    data[:, idx] = distr["type"]( **distr["args"], size=(sample_size,))
random_idx = np.random.choice( np.arange(num_distr), \
                              size=(sample_size,), p=coefficients )
X_samples = data[ np.arange(sample_size), random_idx ]
fig, ax = plt.subplots(1, 1, figsize=(9,3))
ax.hist( X_samples, bins=100, density=True, label='sample', color = 'skyblue' );
ax.set_title('Random sample from mixture of von Mises and Uniform distributions')
# ---------------------------------------------------------------------- #

# ---------------------------------------------------------------------- #
def logLik_2vM( theta, *args ):
    """
    The first derivative of the log-likelihood function 'l' to be set equal to 
    zero in order to estimate the mixture parameters:
        p1, p2: the weights of the two von Mises distributions on semi-circle
        kappa2, kappa2: the concentrations of the two von Mises distributions
        mu1, mu2: the locations of the two von Mises distributions
        theta:= [ p1, kappa1, mu1, p2, kappa2, mu2 ]
        params:= X_samples: the observations sample 
    The function returns a vector F with the derivatives of 'l' wrt the 
        components of theta. 
    This function is to be called with optimize.fsolve function of scipy:
        roots_ = optimize.fsolve( myF_2vM1U, in_guess, args=my_par )
    """
    scal_ = 0.5
    x_ = np.array(args).T
#    print(type(x_))
#    print('x_=', x_.shape)
    
    # the unknown parameters:
    p1_ = theta[0]
    kappa1_ = theta[1]
    m1_ = theta[2]
    p2_ = theta[3]
    kappa2_ = theta[4]
    m2_ = theta[5]
    
    # the 1st von Mises distribution on semi-circle:
    fvm1_ = stats.vonmises.pdf( x_, kappa1_, m1_, scal_ )
    # logvM1 = -np.sum( stats.vonmises.logpdf( x_, kappa1_, m1_, scal_ ) )
    
    # the 2nd von Mises distribution on semi-circle:
    fvm2_ = stats.vonmises.pdf( x_, kappa2_, m2_, scal_ )
    # logvM2 = -np.sum( stats.vonmises.logpdf( x_, kappa2_, m2_, scal_ ) )
        
    # mixture distribution: 
    fm_ = p1_*fvm1_ + p2_*fvm2_
    # logMix = logvM1 + logvM2 + logU - len(x_)*(np.log(p1_*p2_*pu_))
    
    logMix = -np.sum( np.log( fm_ ) )
    
    return logMix
# ---------------------------------------------------------------------- #

in_guess = [ p1_, kappa1_, loc1_, p2_, kappa2_, loc2_ ]
# bound constraints for the variables: 
bnds = ((0., 1.), (0., 100.), (-np.pi/2, np.pi/2), \
        (0., 1.), (0., 100.), (-np.pi/2, np.pi/2))
cons = ({'type': 'eq', 'fun': lambda x:  x[0] + x[3] - 1.0})
results = optimize.minimize(logLik_2vM, in_guess, args=X_samples, \
                            method='SLSQP', bounds=bnds, constraints=cons, \
                            tol=1e-6, options={'maxiter': 100, 'disp': True})
print('METHOD II = ',results.x)
print('-----------------------------------------')
p1_mle, kappa1_mle, mu1_mle, p2_mle, kappa2_mle, mu2_mle = results.x
print('p1, kappa1, mu1, p2, kappa2, mu2 = ',results.x)
res = results.x
data = np.array([[p1_mle, kappa1_mle, mu1_mle],
                 [p2_mle, kappa2_mle, mu2_mle]])

# ---------------------------------------------------------------------- #
# Plot the fit in the same figure as the histogram:
members_list = ["1st von Mises", "2nd von Mises"]
loc_mle = np.array([mu1_mle, mu2_mle])
kap_mle = np.array([kappa1_mle, kappa2_mle])
p_mle = np.array([p1_mle, p2_mle])
dataFrame = pd.DataFrame({'Distribution': members_list, \
                          'Weight': p_mle.ravel(), \
                          'Concentration': kap_mle.ravel(), \
                          'Location': loc_mle.ravel()})
dataFrame = dataFrame[['Distribution', 'Weight', 'Concentration','Location']]
#dataFrame.set_index('Distribution')
print(dataFrame)

# plot in the same histogram the approximations:
x_ = np.linspace( min(X_samples), max(X_samples), N )
X_tot = np.zeros(len(x_),)
for mu, kap, pii in zip(loc_mle, kap_mle, p_mle):
    X_temp = pii*stats.vonmises.pdf( x_, kap, mu, scal_ )
    X_tot += X_temp
    ax.plot(x_, X_temp, linewidth=2, linestyle='--', \
            label=r'$\mu$ = {}, $\kappa$= {}, p= {} '.format(round(mu,3), round(kap,3), round(pii,3)))
ax.plot(x_, X_tot, color='red', linewidth=2, label='fit mixture')
ax.set_xlabel(r'$\theta$ (rad)', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
#ax.legend(loc=2)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=3)

