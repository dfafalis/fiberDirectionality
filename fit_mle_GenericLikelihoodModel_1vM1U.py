#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 16:49:27 2018
Author Dimitrios Fafalis 

This code makes use of the "GenericLikelihoodModel" subclass of the package 
    statsmodels, to fit random variables to a mixture model of one von Mises 
    on the CIRCLE and one Uniform distribution.
    
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


#import dff_StatsTools as dfst

# ---------------------------------------------------------------------- #
# generate random variables from a mixture of von Mises and \
# uniform distributions:
# size of sample: 
N = 1000
# parameters for the von Mises member:
p_ = 0.8 # weight contribution of von Mises 
kappa_ = np.array((10.0)) # concentration for von Mises member 
loc_ = 0.*np.pi/2.0 # location for von Mises member 
loc_cs = np.array(( np.cos(loc_), np.sin(loc_) )) # cos and sin of location
print('loc_cs = ',loc_cs)

#XX_ = stats.vonmises(kappa_, loc_)
#XX_samples = XX_.rvs(N)
#dfst.plot_dist_samples(XX_, XX_samples, title=None, ax=None)
#fig, ax = plt.subplots(1, 1, figsize=(9,3))
#ax.hist( XX_samples, bins=75, density=True );
#ax.set_title('Random sample from mixture of von Mises and Uniform distriburtions')

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

def unif_logpdf( x, loc=xu_1, scale=xu_2 ):
    return stats.uniform.logpdf( x, loc, scale ).sum()

def vonMis_logpdf( x, kappa=kappa_, loc=loc_ ):
    return stats.vonmises.logpdf( x, kappa, loc ).sum()

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
        
        # this is the log-likelihood for a single von Mises: 
        #   -np.log( vonMis_pdf( self.endog, kappa_=kappa1_, loc_=m1_ ) )
        
        # the 1st von Mises distribution:
        fvm1_ = vonMis_logpdf( self.endog, kappa=kappa_, loc=loc_ ) 
        
        # the uniform distribution:
        fu_ = unif_logpdf( self.endog, loc=xu_1, scale=xu_2 ) 
        
        # total log-likelihood: 
        nloglik = p_*fvm1_ + (1 - p_)*fu_
        
        return -nloglik
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        return super( Mix1vonMises1Uniform, self ).fit( start_params=start_params, \
                        maxiter=maxiter, maxfun=maxfun, **kwds)
        
start_params = np.array([ p_, kappa_, loc_ ])
model = Mix1vonMises1Uniform(X_samples)
results = model.fit(start_params)
#print(results.summary())

p_mle, kappa_mle, mu_mle = results.params
print('-----------------------------------------')
print('p, kappa, mu= ',p_mle, kappa_mle, mu_mle)
