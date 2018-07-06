#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 16:49:27 2018
Author Dimitrios Fafalis 

This code makes use of the "GenericLikelihoodModel" subclass of the package 
    statsmodels, to fit random variables to a mixture model of one von Mises 
    on the CIRCLE and one Uniform distributions.
    
    It seems that the optimization algorithm does not take into account
        the uniform distribution: I get the same results even if I ignore
        from the 'nloglikeobs' the uniform distribution! 
    Of course, the result is not correct.
    
    Its generalization to mixtures of von Mises and Uniform distributions has 
        failed, at least to my attempts! 

@author: df
"""

import math
import numpy as np
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
import matplotlib.pyplot as plt
import dff_dispersionCalculator as dC

np.random.seed(123456789)

# ---------------------------------------------------------------------- #
# generate random variables from a mixture of von Mises and \
# uniform distributions:
# size of sample: 
N = 1000
# parameters for the von Mises member:
p_ = 0.60 # weight contribution of von Mises 
kappa_ = np.array((10.0)) # concentration for von Mises member 
loc_ = np.pi/3.0 # location for von Mises member 
loc_cs = np.array(( np.cos(loc_), np.sin(loc_) )) # cos and sin of location
print('loc_cs = ',loc_cs)

# ---------------------------------------------------------------------- #
# now generate the random sample:
# "Creating a mixture of probability distributions for sampling"
# a question on stackoverflow: 
# https://stackoverflow.com/questions/47759577/
#               creating-a-mixture-of-probability-distributions-for-sampling
xu_1 = -np.pi
xu_2 = 2*np.pi
distributions = [
        { "type": stats.vonmises.rvs, "args": {"kappa":kappa_, "loc":loc_ } },
        { "type": stats.uniform.rvs,  "args": {"loc":xu_1, "scale":xu_2} }
]
coefficients = np.array([p_, 1.-p_]) # these are the weights 
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
ax.hist( X_samples, bins=100, density=True );
ax.set_title('Random sample from mixture of von Mises and Uniform distriburtions')
# ---------------------------------------------------------------------- #

# ---------------------------------------------------------------------- #
# some preliminary quantities:
CS = dC.getCosSin( X_samples )
r_ = ( sum(CS[:,0]), sum(CS[:,1]) )
print('r_ =',r_)
r_r = np.sqrt( sum(CS[:,0])**2 + sum(CS[:,1])**2 )
print('r_r =',r_r)
m_bar_cs = r_ / r_r
print('m_bar_cs =',m_bar_cs)
r_bar = r_r / N
print('r_bar =',r_bar)
m_bar = math.atan2( m_bar_cs[1], m_bar_cs[0])
print('m_bar =',m_bar,np.degrees(m_bar))
# for the uniform distribution:
#xu_1 = min(X_samples)
#xu_2 = (max(X_samples) - min(X_samples))
#xu_1 = -np.pi
#xu_2 = 2*np.pi
# ---------------------------------------------------------------------- #

def unif_pdf( x, loc=xu_1, scale=xu_2, weight=p_ ):
    return (1. - p_)*stats.uniform.pdf( x, loc, scale )

def vonMis_pdf( x, kappa=kappa_, loc=loc_, weight=p_ ):
    return p_*stats.vonmises.pdf( x, kappa, loc )

class Mix1vonMises1Uniform(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)
            
        super(Mix1vonMises1Uniform, self).__init__(endog, exog, **kwds)
    
    def nloglikeobs(self, params):
        # the unknown parameters:
        p_ = params[0]
        kappa_ = params[1]
        loc_ = params[2]
        
        # the log-likelihood for the von Mises distribution:
        fvm1_ = np.log( vonMis_pdf( self.endog, kappa=kappa_, loc=loc_, weight=p_ ) )
        
        # the log-likelihood for the uniform distribution:
        fu_ = np.log( unif_pdf( self.endog, loc=xu_1, scale=xu_2, weight=p_ ) )
        
        # total log-likelihood: 
        nloglik = sum(fvm1_ + fu_)

        return -nloglik
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        print('start_params: ', start_params)
        return super( Mix1vonMises1Uniform, self ).fit( start_params=start_params, \
                        maxiter=maxiter, maxfun=maxfun, **kwds)
        
        
model = Mix1vonMises1Uniform(X_samples)
start_params = np.array([ p_, kappa_, loc_ ])
results = model.fit(start_params=start_params)
#print(results.summary())

p_mle, kappa_mle, mu_mle = results.params
print('-----------------------------------------')
print('p, kappa, mu= ',p_mle, kappa_mle, mu_mle)
x = np.linspace(min(X_samples), max(X_samples), N)
Xvm_ = stats.vonmises.pdf( x, kappa_mle, mu_mle )
Xun_ = stats.uniform.pdf( x, xu_1, xu_2 )
Xto_ = p_mle*Xvm_ + (1 - p_mle)*Xun_
ax.plot(x, Xvm_, 'b-', label='von Mises part')
ax.plot(x, Xun_, 'g-', label='uniform part')
ax.plot(x, Xto_, 'r-', label='mix von Mises + Unif fit')
ax.legend()
