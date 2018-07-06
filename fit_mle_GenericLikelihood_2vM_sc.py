#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 20:18:55 2018

@author: df
"""

#import math
import numpy as np
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------- #
# some initial parameters to generate the random sample: 
# size of sample: 
N = 1000
# parameters for the von Mises members:
p1_ = 0.4 # weight contribution of the 1st von Mises 
p2_ = 0.4 # weight contribution of the 1st von Mises 
pu = 1. - p1_ - p2_
kappa1_ = np.array((12.0)) # concentration for the 1st von Mises member 
kappa2_ = np.array((12.0)) # concentration for the 1st von Mises member 
loc1_ = -np.pi/6.0 # location for the 1st von Mises member 
loc2_ =  np.pi/6.0 # location for the 1st von Mises member 
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
        { "type": stats.uniform.rvs,  \
                     "args": {"loc":u1_, "scale":u2_ } }
]
coefficients = np.array( [ p1_, p2_, pu ] ) # these are the weights 
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
ax.set_title('Random sample from mixture of von Mises and Uniform distriburtions')
# ---------------------------------------------------------------------- #

# ---------------------------------------------------------------------- #
def vonMis_logpdf( x, kappa=None, loc=None, scale=None ):
    return stats.vonmises.logpdf( x, kappa, loc, scale ).sum()

def vonMis_pdf( x, kappa=None, loc=None, scale=None ):
    return stats.vonmises.pdf( x, kappa, loc, scale )

class Mix1vonMises1Uniform(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)
            
        super(Mix1vonMises1Uniform, self).__init__(endog, exog, **kwds)
    
    def nloglikeobs(self, params):
        scal_ = 0.5
        # the unknown parameters:
        p1_ = params[0]
        kappa1_ = params[1]
        loc1_ = params[2]
        # p2_ = params[3]
        kappa2_ = params[3]
        loc2_ = params[4]
        p2_ = 1. - p1_
        
        # this is the log-likelihood for a single von Mises: 
        #   -np.log( vonMis_pdf( self.endog, kappa_=kappa1_, loc_=m1_ ) )
        
        # the 1st von Mises distribution:
        fvm1_ = vonMis_pdf( self.endog, kappa=kappa1_, loc=loc1_, scale=scal_ ) 
        
        # the 2nd von Mises distribution:
        fvm2_ = vonMis_pdf( self.endog, kappa=kappa2_, loc=loc2_, scale=scal_ ) 
        
        # mixture distribution: 
        fm_ = p1_*fvm1_ + p2_*fvm2_
        
        # total log-likelihood: 
        nloglik = -np.sum( np.log( fm_ ) )
        
        return nloglik
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        return super( Mix1vonMises1Uniform, self ).fit( start_params=start_params, \
                        maxiter=maxiter, maxfun=maxfun, **kwds)
        
start_params = np.array([ p1_, kappa1_, loc1_, kappa2_, loc2_ ])
model = Mix1vonMises1Uniform(X_samples)
results = model.fit(start_params)
#print(results.summary())
res = results.params

p1_mle, kappa1_mle, mu1_mle, kappa2_mle, mu2_mle = results.params
print('-----------------------------------------')
print('p1, kappa1, mu1, p2, kappa2, mu2 = ',results.params)

# ---------------------------------------------------------------------- #
# Plot the fit in the same figure as the histogram:
members_list = ["1st von Mises", "2nd von Mises"]
loc_mle = np.array([res[2], res[4]])
kap_mle = np.array([res[1], res[3]])
p_mle = np.array([res[0], 1.-res[0]])
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
ax.plot(x_, X_tot, color='red', linewidth=2, linestyle='--', label='fit mixture')
ax.set_xlabel(r'$\theta$ (rad)', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
#ax.legend(loc=2)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=4)
