#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 19:55:11 2018
Author Dimitrios Fafalis 

This code generates a random sample from a mixture of three von Mises on CIRCLE
    with weights p1, p2 and p3 = 1 - p1 - p2, 
    using the functions: "stats.vonmises.rvs" and "stats.uniform.rvs"
    using the "numpy.random.choice" function

This code makes use of "optimize.fsolve" to solve: 
    a. a system of eight (8) nonlinear equations, without jocobian 

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
p1 = 0.25 # weight contribution of the 1st von Mises 
p2 = 0.50 # weight contribution of the 2nd von Mises 
#p3 = 0.3 # weight contribution of the 3rd von Mises 
#pu = 1. - p1 - p2 - p3
p3 = 1. - p1 - p2 # weight contribution of the 3rd von Mises 
kappa1_ = np.array((12.0)) # concentration for the 1st von Mises member 
kappa2_ = np.array((12.0)) # concentration for the 2nd von Mises member 
kappa3_ = np.array((12.0)) # concentration for the 2nd von Mises member 
loc1_ = -np.pi/3.0 # location for the 1st von Mises member 
loc2_ =  0.*np.pi/12.0 # location for the 1st von Mises member 
loc3_ =  np.pi/3.0 # location for the 1st von Mises member 
#loc_cs = np.array(( np.cos(loc_), np.sin(loc_) )) # cos and sin of location
#print('loc_cs = ',loc_cs)
kappas = np.array([kappa1_, kappa2_, kappa3_]) 
locs = np.array([loc1_, loc2_, loc3_])
pis = np.array([p1, p2, p3])
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

u1_ = -np.pi
u2_ = 2*np.pi
distributions = [
        { "type": stats.vonmises.rvs, "args": {"kappa":kappa1_, "loc":loc1_ }},
        { "type": stats.vonmises.rvs, "args": {"kappa":kappa2_, "loc":loc2_ }},
        { "type": stats.vonmises.rvs, "args": {"kappa":kappa3_, "loc":loc3_ }},
]
#        { "type": stats.uniform.rvs,  "args": {"loc":u1_, "scale":u2_ } }
coefficients = np.array( [ p1, p2, p3 ] ) # these are the weights 
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
ax.set_title('Random sample from mixture of three von Mises distriburtions')
# ---------------------------------------------------------------------- #
#qm = X_samples[X_samples < -2]
#qp = X_samples[X_samples >  2]
#qu = np.concatenate([qm,qp])
#X_samples = np.setdiff1d(X_samples, qu)
#ax.hist( X_samples, bins=100, density=True, color='green' );

# ---------------------------------------------------------------------- #
def myF_3vM( theta, *params ):
    """
    The first derivative of the log-likelihood function 'l' to be set equal to 
    zero in order to estimate the mixture parameters:
        p1, p2, p3: the weights of the three von Mises distributions
        kappa1, kappa2, kappa3: the concentrations of von Mises distributions
        mu1, mu2, mu3: the locations of the three von Mises distributions
        theta:= [ p1, kappa1, mu1, p2, kappa2, mu2 ]
        params:= X_samples: the observations sample 
    The function returns a vector F with the derivatives of 'l' wrt the 
    components of theta. 
    This function is to be called with optimize.fsolve function of scipy:
        roots_ = optimize.fsolve( myF_3vM, in_guess, args=my_par )
    """
    x_ = np.array(params).T
    
    # the unknown parameters:
    p1_ = theta[0]
    kappa1_ = theta[1]
    m1_ = theta[2]
    p2_ = theta[3]
    kappa2_ = theta[4]
    m2_ = theta[5]
    kappa3_ = theta[6]
    m3_ = theta[7]
    p3_ = 1.0 - p1_ - p2_
    
    # the 1st von Mises distribution:
    fvm1_ = stats.vonmises.pdf( x_, kappa1_, m1_ )
    # the 2nd von Mises distribution:
    fvm2_ = stats.vonmises.pdf( x_, kappa2_, m2_ )
    # the erd von Mises distribution:
    fvm3_ = stats.vonmises.pdf( x_, kappa3_, m3_ )
        
    # mixture distribution: 
    fm_ = p1_*fvm1_ + p2_*fvm2_ + p3_*fvm3_

    # first derivative wrt weight p1:
    dldp1 = sum( np.divide( np.subtract( fvm1_, fvm3_ ), fm_ ) )
        
    # first derivative wrt weight p2:
    dldp2 = sum( np.divide( np.subtract( fvm2_, fvm3_ ), fm_ ) )
    
    # first derivative wrt location mu1:=
    dldm1 = kappa1_*p1_*sum( np.multiply( np.divide( fvm1_, fm_ ), \
                                         np.sin(x_ - m1_) ) )
    
    # first derivative wrt location mu2:=
    dldm2 = kappa2_*p2_*sum( np.multiply( np.divide( fvm2_, fm_ ), \
                                         np.sin(x_ - m2_) ) )
    
    # first derivative wrt location mu3:=
    dldm3 = kappa3_*p3_*sum( np.multiply( np.divide( fvm3_, fm_ ), \
                                         np.sin(x_ - m3_) ) )
    
    # first derivative wrt concentration kappa1:
    Ak1 = ( iv(1.0, kappa1_) / i0(kappa1_) )    
    dldk1 = p1_*sum( np.multiply( np.divide( fvm1_, fm_ ), \
                                 ( np.cos(x_ - m1_) - Ak1 ) ) )
    
    # first derivative wrt concentration kappa2:
    Ak2 = ( iv(1.0, kappa2_) / i0(kappa2_) )    
    dldk2 = p2_*sum( np.multiply( np.divide( fvm2_, fm_ ), \
                                 ( np.cos(x_ - m2_) - Ak2 ) ) )
    
    # first derivative wrt concentration kappa3:
    Ak3 = ( iv(1.0, kappa3_) / i0(kappa3_) )    
    dldk3 = p3_*sum( np.multiply( np.divide( fvm3_, fm_ ), \
                                 ( np.cos(x_ - m3_) - Ak3 ) ) )

    F = [ dldp1[0], dldk1[0], dldm1[0], \
          dldp2[0], dldk2[0], dldm2[0], \
                    dldk3[0], dldm3[0] ]
    
    return F
# ---------------------------------------------------------------------- #

# obtain the solutions by calling fsolve: 
my_par = X_samples # parameters needed inside F and J 
# initial guesses for the parameters:
in_guess = [ p1, kappa1_, loc1_, p2, kappa2_, loc2_, kappa3_, loc3_ ]
#in_guess = [ 0.5, 11.0, 1.3, 0.5, 6.0, -0.4 ]
print('METHOD II: - - - - - - without providing the Jacobian - - - - - - ')
r_min_III = optimize.fsolve( myF_3vM, in_guess, args=my_par, \
                             full_output=True, xtol=1.49012e-8, maxfev=0 )
print('solution with METHOD II = ', r_min_III)
print('parameters with METHOD II = ', r_min_III[0])
sol_III = r_min_III[0]

kap1 = sol_III[1]
kap2 = sol_III[4]
kap3 = sol_III[6]
mu1 = sol_III[2]
mu2 = sol_III[5]
mu3 = sol_III[7]
pp1 = sol_III[0]
pp2 = sol_III[3]
pp3 = 1. - pp1 - pp2
kapp_pr = np.array([kap1, kap2, kap3])
locs_pr = np.array([mu1, mu2, mu3])
pi_pr = np.array([pp1, pp2, pp3])
# plot in the same histogram the approximations:
x_ = np.linspace( min(X_samples), max(X_samples), N )
X_tot = np.zeros(len(x_),)
for mu, kap, pii in zip(locs_pr, kapp_pr, pi_pr):
    print(mu,kap,pii)
    X_temp = pii*stats.vonmises.pdf( x_, kap, mu )
    ax.plot(x_, X_temp, linewidth=2, \
            label=r'$\mu$ = {}, $\kappa$= {}, p= {} '.format(round(mu,3), round(kap,3), round(pii,3)))
    X_tot += X_temp
ax.plot(x_, X_tot, color='red', linewidth=3, label='fit mixture')
ax.set_xlabel(r'$\theta$ (rad)', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
ax.legend(loc=1)
#ax.legend(loc=1, bbox_to_anchor=(1.1, 1))
