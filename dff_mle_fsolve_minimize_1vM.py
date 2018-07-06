#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:57:10 2018
Author Dimitrios Fafalis 

This code generates a random sample from one von Mises distribution, 
    using the functions: "stats.vonmises(kappa_, loc_)" and "stats.vonmises.rvs"

This code makes use of "optimize.fsolve" to solve: 
    a. one nonlinear equation, without jacobian ( Method Ia ) 
    b. one nonlinear equation, with jacobian ( Method Ib )  
    
This code makes use of "optimize.minimize" to solve: 
    c. one nonlinear equation, without constraints ( Method II ) 
    d. one nonlinear equation, with constraints ( Method III ) 

@author: df
"""

import numpy as np
from scipy import optimize
from scipy import stats
import math
# import cvopt
from numpy import i0 # modified Bessel function of the first kind order 0, I_0
from scipy.special import iv # modified Bessel function of first kind, I-v 
import matplotlib.pyplot as plt

import dff_StatsTools as dfst
import dff_dispersionCalculator as dC

## ---------------------------------------------------------------------- #
## another method, to find parameters of a single von Mises,
## just to check with my codes, 
## is the
#from astropy.stats import vonmisesmle
#from astropy import units as u
#
## in the case my codes here, data is the array of rads (X_samples):
#data = np.array([130, 90, 0, 145])*u.deg
#vonmisesmle(data
## ---------------------------------------------------------------------- #

            
#def main():
#    """
#    main function to test this module before it is incorporated to DF
#    """
    # generate random variables from a von Mises distributions:
N = 1000
kappa_ = np.array((2.0))
loc_ = np.pi/4
loc_cs = np.array(( np.cos(loc_), np.sin(loc_) ))
print(loc_cs)
# X = stats.vonmises.rvs(kappa, loc, size=N)
X = stats.vonmises(kappa_, loc_)
X_samples = X.rvs(N)
dfst.plot_dist_samples(X, X_samples, title=None, ax=None)

# some preliminary quantities:
CS = dC.getCosSin( X_samples )
r_ = ( sum(CS[:,0]), sum(CS[:,1]) )
print('r_ =',r_)
r_r = np.sqrt( sum(CS[:,0])**2 + sum(CS[:,1])**2 )
print('r_r =',r_r)
m_bar_cs = r_ / r_r
print('m_bar_cs =',m_bar_cs)
r_bar = r_r / N
m_bar = math.atan2( m_bar_cs[1], m_bar_cs[0])
print('m_bar =',m_bar,np.degrees(m_bar))
# for 0.53 <= r_bar < 0.85
# k_est = -0.4 + 1.39*r_bar + 0.43/(1 - r_bar)
# print('kappa estimate with formula = ',k_est)
# use fsolve, to estimate 'kappa', when 'mu' has been estimated through
# equation (2.4) of Banerjee(2005):
#r_min = optimize.fsolve(myF_kappa, [0.0], args=r_bar)
## r_min = optimize.fmin_bfgs(myF, (0,0))
#print(r_min)
#
## use fsolve, to estimate 'kappa' and 'mu':
#initParams = [ 1, 1 ]
#results = optimize.minimize(vmf_log2, initParams, method='nelder-mead')
#print(results.x)

# ---------------------------------------------------------------------- #
# define the von Mises with an expression: 
num = 1.0
denom = (2*np.pi) * i0(kappa_)
conss = num / denom

y11 = stats.vonmises.logpdf(X_samples, kappa=kappa_, loc=loc_)
y22 = np.log(conss * np.exp(kappa_ * CS.dot(loc_cs).T))
# plot the logarithm of the pdf of the von Mises distribution sample: 
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(y11, 'r', label='yPred')
ax.plot(y22, 'b', label='yObs')
ax.legend()

sy1 = -np.sum(y11)
sy2 = -np.sum(y22)

    #return X, X_samples

### ---------------------------------------------------------------------- #
## METHOD 0:
#results_0 = stats.vonmises.fit(X_samples)

# ---------------------------------------------------------------------- #
def kappa_initial(r_bar):
    """
    Approximate formulas for kappa
    """
    if r_bar < 0.53:
        kappa_0 = 2*r_bar + r_bar**3 + (5./6.)*r_bar**5
    elif r_bar >= 0.53 and r_bar < 0.85:
        kappa_0 = -0.4 + 1.39*r_bar + 0.43/(1 - r_bar)
    else:
        kappa_0 = 1./( r_bar**3 -4*r_bar**2 + 3*r_bar )
        
    return kappa_0
# ---------------------------------------------------------------------- #

## ---------------------------------------------------------------------- #
# METHOD I:
# use the 'fsolve' function of the optimize module,
#   to find the root to equation (2.5) of Banerjee(2005)
#   that gives an estimate of the concentration kappa.
print('METHOD I: - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
def myF_kappa(kappa, *params):
    """
    Equation (2.5) of Banerjee(2005)
    I1(kappa)/I0(kappa) = r_bar
    to estimate the concentration 'kappa' of one single von Mises distribution
    The estimation of the location 'mu' is incorporated to the term r_bar
    kappa = theta[0]
    """
    r_ = params[0]
    n_ = params[1]
    F = r_ - n_*( iv(1.0, kappa) / i0(kappa) )
    
    return F


def myJ_kappa(kappa, *params):
    """
    The derivative of U_vm(kappa) = R - n A(kappa) = 0 
    kappa = theta[0]
    """
    N = params[1]
    H = N*( np.power( ( iv(1.0, kappa) / i0(kappa) ), 2.0 ) + \
                      ( iv(1.0, kappa) / i0(kappa) )/kappa - 1.0 )
    return H
    

my_par = ( r_r, N )
print('METHOD Ia: - - - - - - without providing the Jacobian - - - - - - ')
r_min = optimize.fsolve( myF_kappa, kappa_initial(r_bar), args=my_par, \
                         full_output=True, xtol=1.49012e-8, maxfev=0 )
print('kappa with METHOD Ia = ',r_min)
print('METHOD Ib: - - - - - - providing the Jacobian - - - - - - - - - - ')
r_min_H = optimize.fsolve( myF_kappa, kappa_initial(r_bar), fprime=myJ_kappa, \
                    args=my_par, full_output=True, xtol=1.49012e-8, maxfev=0 )
print('kappa with METHOD Ib = ',r_min_H)
print('# -------------------------------------------------------------- #')
## ---------------------------------------------------------------------- #        
    
## ---------------------------------------------------------------------- #
# METHOD II:
# use the 'minimize' function of the optimize module,
#   to estimate 'kappa' and 'mu', using the log-likelihood function
#   and the stats.vonmises functionality of the stats package.
print('METHOD II: - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
def vmf_log2(theta):
    """
    Computes the log(vM(X, kappa, mu) )using the "stats.vonmises.logpdf" 
    X_samples: r.v. from a von Mises distribution; rads 
    """
    kappa = theta[0]
    mu = theta[1]
    
    # Calculate the negative log-likelihood as the negative sum of the log 
    # of the PDF of a von Mises distributions, with location mu and 
    # concentration kappa: 
    logLik = -np.sum( stats.vonmises.logpdf( X_samples, kappa=kappa, loc=mu ) )
        
    return logLik


initParams_2 = np.array([ kappa_initial(r_bar), m_bar ], dtype=np.float64)
results_2 = optimize.minimize(vmf_log2, initParams_2, method='nelder-mead', \
                              tol=1e-6, \
                              options={'maxiter': 100, 'disp': True})
print('METHOD II = ',results_2.x)
print('kappa with METHOD II = ',results_2.x[0])
mu_loglik_2 = results_2.x[1]
print('location with METHOD II (rad), exact = ',mu_loglik_2,loc_)
loc_loglik_2 = np.array(( np.cos(mu_loglik_2), np.sin(mu_loglik_2) ))
print('location with METHOD II (CS) = ',loc_loglik_2)
print('# -------------------------------------------------------------- #')
## ---------------------------------------------------------------------- #

## ---------------------------------------------------------------------- #
# METHOD III:
# use the 'minimize' function of the optimize module,
#   to estimate 'kappa' and 'mu', using the log-likelihood function
#   and the stats.vonmises functionality of the stats package.
#   under the constraint mu.mu = 1.
print('METHOD III: - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
def vmf_log3(params3):
    """
    Computes the  log(vM(X, kappa, mu) )using built-in numpy/scipy Bessel 
    approximations:
        log( c(kappa) * exp(kappa * mu * X) )
    CS: array with the cosine and sine of the r.v. from a von Mises. 
    """
    kappa = params3[0]
    mu = np.array(( params3[1], params3[2] ))
    
    num = 1.0
    denom = (2*np.pi) * i0(kappa)
    conss = num / denom
    
    logLik = -np.sum( np.log( conss * np.exp(kappa * CS.dot(mu).T) ) )
    
    return logLik


# use minimize, to estimate 'kappa' and 'mu' under the constraint mu.mu = 1:
def g(params3):
    """
    Constraint mu.(mu).T = 1
    """
    mu = np.array(( params3[1], params3[2] ))
    return mu.dot(mu).T - 1


initParams_3 = np.array([ kappa_initial(r_bar), m_bar_cs[0], m_bar_cs[1] ], \
                         dtype=np.float64)
constraint = dict(type='eq', fun=g)
results_3 = optimize.minimize(vmf_log3, initParams_3, method='SLSQP', \
                              constraints=[constraint], \
                              tol=1e-6, \
                              options={'maxiter': 100, 'disp': True})
print('METHOD III = ',results_3.x)
print('kappa with METHOD III  = ',results_3.x[0])
print('pred_loc_cs_3 = ',results_3.x[1:])
temp = math.atan2( results_3.x[2], results_3.x[1])
print('pred_loc_rad_3 = ',temp,loc_)
print('# -------------------------------------------------------------- #')
## ---------------------------------------------------------------------- #
print('the crude initial estimate for kappa is = ',kappa_initial(r_bar))

def _vmf_log(X, kappa, mu):
    """
    Computes the  log(vM(X, kappa, mu) )using built-in numpy/scipy Bessel 
    approximations:
        log( c(kappa) * exp(kappa * mu * X) )
    """
    # n = X.shape
    return np.log(_vmf_normalize(kappa) * np.exp(kappa * X.dot(mu).T))


def _vmf_normalize(kappa):
    """
    Compute normalization constant using built-in numpy/scipy Bessel 
    approximations:
        c(kappa) = 1 / (2 * pi * I0(kappa))
    """
    num = 1.0
    denom = (2*np.pi) * i0(kappa)
    
    return num / denom

