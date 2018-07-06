#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 19:55:11 2018
Author Dimitrios Fafalis 

This code generates a random sample from a mixture of two von Mises on CIRCLE
    with weights p and (1-p) using the functions: 
    "stats.vonmises.rvs" and the "numpy.random.choice" function

This code makes use of "optimize.fsolve" to solve: 
    a. a system of five (5) nonlinear equations, without jocobian 

@author: df
"""

import numpy as np
from scipy import optimize
from scipy import stats
import math
from numpy import i0 # modified Bessel function of the first kind order 0, I_0
from scipy.special import iv # modified Bessel function of first kind, I-v 
import matplotlib.pyplot as plt

#import dff_StatsTools as dfst
#import dff_dispersionCalculator as dC

# size of sample: 
N = 1000
# parameters for the von Mises member:
p1 = 0.5 # weight contribution of the 1st von Mises 
p2 = 1. - p1 # weight contribution of the 2nd von Mises 
kappa1_ = np.array((12.0)) # concentration for the 1st von Mises member 
kappa2_ = np.array((5.0)) # concentration for the 2nd von Mises member 
loc1_ = -np.pi/6.0 # location for the 1st von Mises member 
loc2_ =  np.pi/6.0 # location for the 1st von Mises member 
#loc_cs = np.array(( np.cos(loc_), np.sin(loc_) )) # cos and sin of location
#print('loc_cs = ',loc_cs)
kappas = np.array([kappa1_, kappa2_]) 
locs = np.array([loc1_, loc2_])
pis = np.array([p1, p2])
## ---------------------------------------------------------------------- #
#def _vmf_rvs(kappa, loc, size=N):
#    """
#    Simulates n random angles from a von Mises distribution
#    with preferred direction loc and concentration kappa. 
#    """
#    a = 1. + np.sqrt(1. + 4.*kappa**2)
#    b = (a - np.sqrt(2.*a))/(2.*kappa)
#    r = (1. + b**2)/(2.*b)
#    
#    theta = np.zeros(N)
#    for j in np.arange(N):
#        while True:
#            u = np.random.rand(3)
#            
#            z = np.cos(np.pi*u[0])
#            f = (1. + r*z)/(r + z)
#            c = kappa*(r - f)
#            
#            if u[1] < c*(2. - c) or np.log(c) - np.log(u[1]) + 1. - c < 0:
#                break
#    
#        theta[j] = loc + np.sign(u[2] - 0.5)*np.arccos(f)
#    
#    return theta
#
#
## now generate the random sample:
#distributions = [
#        { "type": _vmf_rvs, "args": {"kappa":kappa1_, "loc":loc1_ }},
#        { "type": _vmf_rvs, "args": {"kappa":kappa2_, "loc":loc2_ }}
#]
## ---------------------------------------------------------------------- #

# ---------------------------------------------------------------------- #
# now generate the random sample:
# "Creating a mixture of probability distributions for sampling"
# a question on stackoverflow: 
# https://stackoverflow.com/questions/47759577/
#               creating-a-mixture-of-probability-distributions-for-sampling


distributions = [
        { "type": stats.vonmises.rvs, "args": {"kappa":kappa1_, "loc":loc1_ }},
        { "type": stats.vonmises.rvs, "args": {"kappa":kappa2_, "loc":loc2_ }}
]
coefficients = np.array( [ p1, p2 ] ) # these are the weights 
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
ax.hist( X_samples, bins=100, density=True, label='sample', color = 'skyblue');
ax.set_title('Random sample from mixture of two von Mises distriburtions')
# ---------------------------------------------------------------------- #
#qm = X_samples[X_samples < -2]
#qp = X_samples[X_samples >  2]
#qu = np.concatenate([qm,qp])
#X_samples = np.setdiff1d(X_samples, qu)
#ax.hist( X_samples, bins=100, density=True, color='green' );

# ---------------------------------------------------------------------- #
def myF_2vM( theta, *params ):
    """
    The first derivative of the log-likelihood function 'l' to be set equal to 
    zero in order to estimate the mixture parameters:
        p1, p2: the weights of the two von Mises distributions
        kappa1, kappa2: the concentrations of the two von Mises distributions
        mu1, mu2: the locations of the two von Mises distributions
        theta:= [ p1, kappa1, mu1, kappa2, mu2 ]
        params:= X_samples: the observations sample 
    The function returns a vector F with the derivatives of 'l' wrt the 
    components of theta. 
    This function is to be called with optimize.fsolve function of scipy:
        roots_ = optimize.fsolve( myF_2vM, in_guess, args=my_par )
    """
    x_ = np.array(params).T
#    print(type(x_))
#    print('x_=', x_.shape)
    
    # the unknown parameters:
    p1_ = theta[0]
    kappa1_ = theta[1]
    m1_ = theta[2]
    kappa2_ = theta[3]
    m2_ = theta[4]
    p2_ = 1.0 - p1_ 
    
    # the 1st von Mises distribution:
    fvm1_ = stats.vonmises.pdf( x_, kappa1_, m1_ )
#    print('fvm_=', fvm_.shape)
    # the 2nd von Mises distribution:
    fvm2_ = stats.vonmises.pdf( x_, kappa2_, m2_ )
        
    # mixture distribution: 
    fm_ = p1_*fvm1_ + p2_*fvm2_ 
#    print('fm_=', fm_.shape)

    # first derivative wrt weight p1:
    dldp1 = sum( np.divide( np.subtract( fvm1_, fvm2_ ), fm_ ) )
#    print('dldp=', dldp.shape)
        
    # first derivative wrt location mu1:=
    dldm1 = kappa1_*p1_*sum( np.multiply( np.divide( fvm1_, fm_ ), \
                                         np.sin(x_ - m1_) ) )
#    print('dldm=', dldm.shape)
    
    # first derivative wrt location mu2:=
    dldm2 = kappa2_*p2_*sum( np.multiply( np.divide( fvm2_, fm_ ), \
                                         np.sin(x_ - m2_) ) )
    
    # first derivative wrt concentration kappa1:
    Ak1 = ( iv(1.0, kappa1_) / i0(kappa1_) )    
    dldk1 = p1_*sum( np.multiply( np.divide( fvm1_, fm_ ), \
                                 ( np.cos(x_ - m1_) - Ak1 ) ) )
#    print('dldk=', dldk.shape)
    
    # first derivative wrt concentration kappa1:
    Ak2 = ( iv(1.0, kappa2_) / i0(kappa2_) )    
    dldk2 = p2_*sum( np.multiply( np.divide( fvm2_, fm_ ), \
                                 ( np.cos(x_ - m2_) - Ak2 ) ) )
    
    F = [ dldp1[0], dldk1[0], dldm1[0], dldk2[0], dldm2[0] ]
    
    return F
# ---------------------------------------------------------------------- #

# obtain the solutions by calling fsolve: 
my_par = X_samples # parameters needed inside F and J 
# initial guesses for the parameters:
in_guess = [ p1, kappa1_, loc1_, kappa2_, loc2_ ]
#in_guess = [ 0.5, 11.0, 1.3, 0.5, 6.0, -0.4 ]
print('METHOD II: - - - - - - without providing the Jacobian - - - - - - ')
r_min_III = optimize.fsolve( myF_2vM, in_guess, args=my_par, \
                             full_output=True, xtol=1.49012e-8, maxfev=0 )
print('solution with METHOD II = ', r_min_III)
print('parameters with METHOD II = ', r_min_III[0])
sol_III = r_min_III[0]

locs_pr = np.array([sol_III[2],sol_III[4]])
kapp_pr = np.array([sol_III[1],sol_III[3]])
pi_pr = np.array([sol_III[0], 1.-sol_III[0]])
# plot in the same histogram the approximations:
x_ = np.linspace( min(X_samples), max(X_samples), N )
X_tot = np.zeros(len(x_),)
for mu, kap, pii in zip(locs_pr, kapp_pr, pi_pr):
    X_temp = pii*stats.vonmises.pdf( x_, kap, mu )
    X_tot += X_temp
    ax.plot(x_, X_temp, linewidth=2, \
            label=r'$\mu$ = {}, $\kappa$= {}, p= {} '.format(round(mu,3), round(kap,3), round(pii,3)))
ax.plot(x_, X_tot, 'r:', linewidth=3, label='fit mixture')
ax.set_xlabel(r'$\theta$ (rad)', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
ax.legend(loc=0)
