#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 19:55:11 2018
Author Dimitrios Fafalis 

This code generates a random sample from a mixture of two von Mises 
    on the SEMI-CIRCLE and one uniform distribution, 
    with weights p1, p2 and (1-p1-p2) 
    using the functions: "stats.vonmises.rvs" and "stats.uniform.rvs"
    using the "numpy.random.choice" function

This code makes use of "optimize.fsolve" to solve: 
    a. a system of six (6) nonlinear equations, without jocobian 
    
The solution includes the parameters of the two von Mises and the weights. 

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
p1 = 0.4 # weight contribution of the 1st von Mises 
p2 = 0.4 # weight contribution of the 1st von Mises 
pu = 1. - p1 - p2 # weight contribution of the uniform 
kappa1_ = np.array((5.0)) # concentration for the 1st von Mises member 
kappa2_ = np.array((5.0)) # concentration for the 1st von Mises member 
loc1_ = -np.pi/3.0 # location for the 1st von Mises member 
loc2_ =  np.pi/3.0 # location for the 1st von Mises member 
#loc_cs = np.array(( np.cos(loc_), np.sin(loc_) )) # cos and sin of location
#print('loc_cs = ',loc_cs)
kappas = np.array([kappa1_, kappa2_]) 
locs = np.array([loc1_, loc2_])
pis = np.array([p1, p2, pu])

# parameter to define the user-defined von Mises on the SEMI-CIRCLE 
ll = 2.0
# parameter to define the stats von Mises on the SEMI-CIRCLE 
scal_ = 0.5

# ---------------------------------------------------------------------- #
# now generate the random sample:
# "Creating a mixture of probability distributions for sampling"
# a question on stackoverflow: 
# https://stackoverflow.com/questions/47759577/
#               creating-a-mixture-of-probability-distributions-for-sampling
u1_ = -np.pi/2
u2_ =  np.pi
distributions = [
        { "type": stats.vonmises.rvs, "args": {"kappa":kappa1_, "loc":loc1_, "scale":scal_ }},
        { "type": stats.vonmises.rvs, "args": {"kappa":kappa2_, "loc":loc2_, "scale":scal_ }},
        { "type": stats.uniform.rvs,  "args": {"loc":u1_, "scale":u2_ } }
]
coefficients = np.array( [ p1, p2, pu ] ) # these are the weights 
coefficients /= coefficients.sum() # in case these did not add up to 1
sample_size = N

num_distr = len(distributions)
data = np.zeros((sample_size, num_distr))
for idx, distr in enumerate(distributions):
    data[:, idx] = distr["type"]( **distr["args"], size=(sample_size,))
random_idx = np.random.choice( np.arange(num_distr), \
                              size=(sample_size,), p=coefficients )
X_samples = data[ np.arange(sample_size), random_idx ]
fig, ax = plt.subplots(1, 1, figsize=(10,3))
ax.hist( X_samples, bins=100, density=True, label='sample', color = 'skyblue' );
ax.set_title('Random sample from mixture of von Mises and Uniform distriburtions')
# ---------------------------------------------------------------------- #


# ---------------------------------------------------------------------- #
def myF_2vM1U( theta, *params ):
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
    x_ = np.array(params).T
#    print(type(x_))
#    print('x_=', x_.shape)
    
    # the unknown parameters:
    p1_ = theta[0]
    kappa1_ = theta[1]
    m1_ = theta[2]
    p2_ = theta[3]
    kappa2_ = theta[4]
    m2_ = theta[5]
    pu_ = 1.0 - p1_ - p2_
    
    # the 1st von Mises distribution on semi-circle:
    fvm1_ = stats.vonmises.pdf( x_, kappa1_, m1_, scal_ )
#    print('fvm_=', fvm_.shape)
    # the 2nd von Mises distribution on semi-circle:
    fvm2_ = stats.vonmises.pdf( x_, kappa2_, m2_, scal_ )
    
    # the uniform distribution:
    xu_1 = min(x_)
    xu_2 = (max(x_) - min(x_))
    fu_ = stats.uniform.pdf( x_, loc=xu_1, scale=xu_2 )
#    print('fu_=', fu_.shape)
    
    # mixture distribution: 
    fm_ = p1_*fvm1_ + p2_*fvm2_ + pu_*fu_
#    print('fm_=', fm_.shape)

    # first derivative wrt weight p1:
    dldp1 = sum( np.divide( np.subtract( fvm1_, fu_ ), fm_ ) )
#    print('dldp=', dldp.shape)
    
    # first derivative wrt weight p2:
    dldp2 = sum( np.divide( np.subtract( fvm2_, fu_ ), fm_ ) )
    
    # first derivative wrt location mu1:=
    dldm1 = (kappa1_*p1_/scal_)*sum( np.multiply( np.divide( fvm1_, fm_ ), \
                                         np.sin((x_ - m1_)/scal_) ) )
#    print('dldm=', dldm.shape)
    
    # first derivative wrt location mu2:=
    dldm2 = (kappa2_*p2_/scal_)*sum( np.multiply( np.divide( fvm2_, fm_ ), \
                                         np.sin((x_ - m2_)/scal_) ) )
    
    # first derivative wrt concentration kappa1:
    Ak1 = ( iv(1.0, kappa1_) / i0(kappa1_) )    
    dldk1 = p1_*sum( np.multiply( np.divide( fvm1_, fm_ ), \
                                 ( np.cos((x_ - m1_)/scal_) - Ak1 ) ) )
#    print('dldk=', dldk.shape)
    
    # first derivative wrt concentration kappa1:
    Ak2 = ( iv(1.0, kappa2_) / i0(kappa2_) )    
    dldk2 = p2_*sum( np.multiply( np.divide( fvm2_, fm_ ), \
                                 ( np.cos((x_ - m2_)/scal_) - Ak2 ) ) )
    
    F = [ dldp1[0], dldk1[0], dldm1[0], dldp2[0], dldk2[0], dldm2[0] ]
    
    return F
# ---------------------------------------------------------------------- #

# obtain the solutions by calling fsolve: 
my_par = X_samples # parameters needed inside F and J 
# initial guesses for the parameters:
in_guess = [ p1, kappa1_, loc1_, p2, kappa2_, loc2_ ]
#in_guess = [ 0.5, 11.0, 1.3, 0.5, 6.0, -0.4 ]
print('METHOD II: - - - - - - without providing the Jacobian - - - - - - ')
r_min_III = optimize.fsolve( myF_2vM1U, in_guess, args=my_par, \
                             full_output=True, xtol=1.49012e-8, maxfev=0 )
print('solution with METHOD II = ', r_min_III)
print('parameters with METHOD II = ', r_min_III[0])
sol_III = r_min_III[0]

locs_pr = np.array([sol_III[2],sol_III[5]])
kapp_pr = np.array([sol_III[1],sol_III[4]])
pi_pr = np.array([sol_III[0], sol_III[3]])
# plot in the same histogram the approximations:
x_ = np.linspace( min(X_samples), max(X_samples), N )
X_tot = np.zeros(len(x_),)
for mu, kap, pii in zip(locs_pr, kapp_pr, pi_pr):
    X_temp = pii*stats.vonmises.pdf( x_, kap, mu, scal_ )
    X_tot += X_temp
    ax.plot(x_, X_temp, linewidth=2, linestyle='--', \
            label=r'$\mu$ = {}, $\kappa$= {}, p= {} '.format(round(mu,3), round(kap,3), round(pii,3)))

pu_pr = 1.-sol_III[0]-sol_III[3]
xu_1 = min(X_samples)
xu_2 = max(X_samples) - min(X_samples)
X_un = pu_pr*stats.uniform.pdf( x_, loc=xu_1, scale=xu_2 )
X_tot += X_un
ax.plot(x_, X_un, linewidth=2, label='uniform distr, p= {} '.format(round(pu_pr,3)))
ax.plot(x_, X_tot, color='red', linewidth=2, label='fit mixture')
ax.set_xlabel(r'$\theta$ (rad)', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
#ax.legend(loc=2)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=3)
