#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 16:49:27 2018
Author Dimitrios Fafalis 

This code makes use of the "GenericLikelihoodModel" subclass of the package 
    statsmodels, to fit random variables to a single von Mises distributions.
    
    It works quite well!

    This version is using the function 
        "stats.vonmises.logpdf( x, kappa, loc ).sum()"

@author: df
"""

import numpy as np
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel

N = 1000
kappa_ = np.array((12.0))
loc_ = np.pi/4
loc_cs = np.array(( np.cos(loc_), np.sin(loc_) ))
print(loc_cs)
# X = stats.vonmises.rvs(kappa, loc, size=N)
X = stats.vonmises(kappa_, loc_)
X_samples = X.rvs(N)


def vonMis_logpdf( x, kappa=kappa_, loc=loc_ ):
    return stats.vonmises.logpdf( x, kappa, loc ).sum()


class SinglevonMises(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)
            
        super(SinglevonMises, self).__init__(endog, exog, **kwds)
    
    def nloglikeobs(self, params):
        kappa_ = params[0]
        loc_ = params[1]
        ll = vonMis_logpdf( self.endog, kappa=kappa_, loc=loc_ )
        
        return -ll
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        return super( SinglevonMises, self ).fit( start_params=start_params, \
                        maxiter=maxiter, maxfun=maxfun, **kwds)
        
start_params = np.array([kappa_, loc_])
model = SinglevonMises(X_samples)
results = model.fit(start_params)
print(results.summary())

kappa_mle, mu_mle = results.params

print('kappa, mu= ',kappa_mle, mu_mle)

#print(model.summary())
#ci = model.conf_int(alpha=.05)

