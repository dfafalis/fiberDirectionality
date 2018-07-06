#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:33:10 2018
Author Dimitrios Fafalis 

This code generates a random sample from a mixture of one von Mises on CIRCLE 
    and one uniform distribution, with weights p and (1-p) 
    using the functions: "stats.vonmises.rvs" and "stats.uniform.rvs"
    using the "numpy.random.choice" function

This code makes use of "optimize.fsolve" to solve: 
    a. one nonlinear equation, without jacobian ( Method Ia ) 
    b. one nonlinear equation, with jacobian ( Method Ib )  
    c. a system of three nonlinear equations, without jocobian ( Method II ) 

@author: df
"""

import numpy as np
from scipy import optimize
from scipy import stats
import math
from numpy import i0 # modified Bessel function of the first kind order 0, I_0
from scipy.special import iv # modified Bessel function of first kind, I-v 
import matplotlib.pyplot as plt

import dff_StatsTools as dfst
import dff_dispersionCalculator as dC

# ---------------------------------------------------------------------- #
# generate random variables from a mixture of von Mises and \
# uniform distributions:
# size of sample: 
N = 1000
# parameters for the von Mises member:
p = 0.6 # weight contribution of von Mises 
kappa_ = np.array((10.0)) # concentration for von Mises member 
loc_ = np.pi/3.0 # location for von Mises member 
loc_cs = np.array(( np.cos(loc_), np.sin(loc_) )) # cos and sin of location
print('loc_cs = ',loc_cs)

# ---------------------------------------------------------------------- #
# given the previous von Mises parameters, create and plot the PDF for x: 
x_ = np.linspace( stats.vonmises.ppf(0.001, kappa_, loc_ ), \
                  stats.vonmises.ppf(0.999, kappa_, loc_ ), N )
Xvm_ = stats.vonmises.pdf(x_, kappa_, loc_ )
fig, ax = plt.subplots(1, 1, figsize=(9,3))
ax.set_title('von Mises and uniform distributions with different weights')
ax.plot(x_, Xvm_, 'r-', lw=5, alpha=0.6, label='vonmises pdf')
ax.plot(x_, p*Xvm_, 'b-', lw=5, alpha=0.6, label='vonmises pdf')
ax.plot(x_, (1.-p)*Xvm_, 'g-', lw=5, alpha=0.6, label='vonmises pdf')

# for the uniform distribution:
xu_1 = min(x_)
xu_2 = (max(x_)-min(x_))
xu = np.linspace( stats.uniform.ppf( 0.01, loc=xu_1, scale=xu_2 ), \
                  stats.uniform.ppf( 0.99, loc=xu_1, scale=xu_2 ), N)
Xu_ = stats.uniform.pdf(xu, loc=xu_1, scale=xu_2 )
ax.plot(xu, Xu_, 'r-.', lw=5, alpha=0.6, label='uniform pdf')
ax.plot(xu, p*Xu_, 'b-.', lw=5, alpha=0.6, label='uniform pdf')
ax.plot(xu, (1.-p)*Xu_, 'g-.', lw=5, alpha=0.6, label='uniform pdf')
ax.legend()
# ---------------------------------------------------------------------- #

# ---------------------------------------------------------------------- #
# now generate the random sample:
# "Creating a mixture of probability distributions for sampling"
# a question on stackoverflow: 
# https://stackoverflow.com/questions/47759577/
#               creating-a-mixture-of-probability-distributions-for-sampling
distributions = [
        { "type": stats.vonmises.rvs, "args": {"kappa":kappa_, "loc":loc_ } },
        { "type": stats.uniform.rvs,  "args": {"loc":-np.pi, "scale":2*np.pi} }
]
coefficients = np.array([p, 1.-p]) # these are the weights 
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
# ---------------------------------------------------------------------- #

# ---------------------------------------------------------------------- #
def kappa_initial(r_bar):
    """
    Approximate formulas for kappa;
    Can serve as initial values for the Newton-Raphson method 
    """
    if r_bar < 0.53:
        kappa_0 = 2*r_bar + r_bar**3 + (5./6.)*r_bar**5
    elif r_bar >= 0.53 and r_bar < 0.85:
        kappa_0 = -0.4 + 1.39*r_bar + 0.43/(1 - r_bar)
    else:
        kappa_0 = 1./( r_bar**3 -4*r_bar**2 + 3*r_bar )
        
    return kappa_0
# ---------------------------------------------------------------------- #

# ---------------------------------------------------------------------- #
print('METHOD I: - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
# the functions that contain the nonlinear equation and its jacobian:
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


# obtain the solutions by calling fsolve: 
my_par = ( r_r, N ) # parameters needed inside F and J 
print('METHOD Ia: - - - - - - without the Jacobian - - - - - - -  - ')
r_min = optimize.fsolve( myF_kappa, kappa_initial(r_bar), args=my_par, \
                         full_output=True, xtol=1.49012e-8, maxfev=0 )
print('solution with METHOD Ia = ', r_min)
print('kappa with METHOD Ia = ', r_min[0])

print('METHOD Ib: - - - - - - with the Jacobian - - - - - - - - - - ')
r_min_H = optimize.fsolve( myF_kappa, kappa_initial(r_bar), fprime=myJ_kappa, \
                    args=my_par, full_output=True, xtol=1.49012e-8, maxfev=0 )
print('solution with METHOD Ib = ',r_min_H)
print('kappa with METHOD Ib = ',r_min_H[0])
print('# -------------------------------------------------------------- #')
      
# plot the PDF along with the sample: 
x = np.linspace(min(X_samples), max(X_samples), N)
Xvm_a = stats.vonmises.pdf( x, r_min[0], m_bar )
Xvm_b = stats.vonmises.pdf( x, r_min_H[0], m_bar )
fig, ax = plt.subplots(1, 1, figsize=(9,3))
ax.hist( X_samples, bins=100, density=True, label='sample', color = "skyblue" );
ax.set_title('Random sample from mixture of von Mises and Uniform distriburtions')
ax.plot(x, Xvm_a, 'r-', label='von Mises fit a')
ax.plot(x, Xvm_b, 'g:', label='von Mises fit b')
ax.legend()
fig_name = 'Fig_random_sample_mix_vM_Un'
fig.savefig(fig_name + '.eps')
# ---------------------------------------------------------------------- #

xxx = stats.vonmises.pdf( X_samples, r_min[0], m_bar )


p0_ = p
mu0_ = m_bar
kappa0_ = r_min_H[0]
# ---------------------------------------------------------------------- #
# now program how to solve for the mixture unknown parameters:
# the weight p, and the von Mises location mu and concentration kappa 
def der_p( x_, kappa_, m_, p_ ):
    """
    Evaluate the first derivative of the log-likelihood function l 
    at p=1. This step is required to assess whether the mixture is purely 
    von Mises or not.
    """
    print('xshape=',x_.shape)
    # the von Mises distribution:
    fvm_ = stats.vonmises.pdf( x_, kappa_, m_ )
    print(fvm_.shape)
    # the uniform distribution:
    xu_1 = min(x_)
    xu_2 = (max(x_) - min(x_))
    fu_ = stats.uniform.pdf( x_, loc=xu_1, scale=xu_2 )
    print(fu_.shape)
    # mixture distribution: 
    fm_ = p_*fvm_ + (1. - p_)*fu_
    print(fm_.shape)
    tt = (fvm_ - fu_)/fm_
    print(tt.shape)
    sum_tt = sum(tt)
    print(sum_tt)
    
    return sum( (fvm_ - fu_)/fm_ )
    
# ---------------------------------------------------------------------- #

# check whether the first derivativ of the log-likelihood function wrt weight
# p evaluated at p=1 is positive:
dldp1 = der_p( X_samples, kappa0_, mu0_, 1.0 )
print('first der l wrt p at p=1, = ', dldp1)

# ---------------------------------------------------------------------- #
def myF_1vM1U( theta, *params ):
    """
    The first derivative of the log-likelihood function 'l' to be set equal to 
    zero in order to estimate the mixture parameters:
        p1: the weight of the von Mises distribution 
        kappa1: the concentration of the von Mises distribution 
        mu1: the location of the von Mises distribution 
        theta:= [ p1, kappa1, mu1 ]
        params:= X_samples: the observations sample 
    The function returns a vector F with the derivatives of 'l' wrt the 
    components of theta. 
    This function is to be called with optimize.fsolve function of scipy:
        roots_ = optimize.fsolve( myF_1vM1U, in_guess, args=my_par )
    """
    x_ = np.array(params).T
#    print(type(x_))
#    print('x_=', x_.shape)
    
    # the unknown parameters:
    p_ = theta[0]
    kappa_ = theta[1]
    m_ = theta[2]
    
    # the von Mises distribution:
    fvm_ = stats.vonmises.pdf( x_, kappa_, m_ )
#    print('fvm_=', fvm_.shape)
    # the uniform distribution:
    xu_1 = min(x_)
    xu_2 = (max(x_) - min(x_))
    fu_ = stats.uniform.pdf( x_, loc=xu_1, scale=xu_2 )
#    print('fu_=', fu_.shape)
    # mixture distribution: 
    fm_ = p_*fvm_ + (1. - p_)*fu_
#    print('fm_=', fm_.shape)

    # first derivative wrt weight p:
    dldp = sum( np.divide( np.subtract( fvm_, fu_ ), fm_ ) )
#    print('dldp=', dldp.shape)
    
    # first derivative wrt location mu:=
    dldm = kappa_*p_*sum( np.multiply( np.divide( fvm_, fm_ ), np.sin(x_ - m_) ) )
#    print('dldm=', dldm.shape)
    
    # first derivative wrt concentration kappa:
    Ak = ( iv(1.0, kappa_) / i0(kappa_) )    
    dldk = p_*sum( np.multiply( np.divide( fvm_, fm_ ), ( np.cos(x_ - m_) - Ak ) ) )
#    print('dldk=', dldk.shape)
    
    F = [ dldp[0], dldk[0], dldm[0] ]
    
    return F
# ---------------------------------------------------------------------- #

# obtain the solutions by calling fsolve: 
my_par = X_samples # parameters needed inside F and J 
# initial guesses for the parameters:
in_guess = [ p0_, kappa0_, mu0_ ]
print('METHOD II: - - - - - - without providing the Jacobian - - - - - - ')
r_min_II = optimize.fsolve( myF_1vM1U, in_guess, args=my_par, \
                         full_output=True, xtol=1.49012e-8, maxfev=0 )
print('solution with METHOD II = ', r_min_II)
print('parameters with METHOD II = ', r_min_II[0])
sol_II = r_min_II[0]

Xvm_m = stats.vonmises.pdf( x, sol_II[1], sol_II[2] )
xum_1 = min(x)
xum_2 = (max(x)-min(x))
Xu_m = stats.uniform.pdf( x, loc=xum_1, scale=xum_2 )
X_m = sol_II[0]*Xvm_m + (1. - sol_II[0])*Xu_m

fig, ax = plt.subplots(1, 1, figsize=(9,3))
ax.hist( X_samples, bins=100, density=True, label='sample', color = "skyblue" );
ax.set_title('Random sample from mixture of von Mises and Uniform distriburtions')
ax.plot(x, Xvm_m, 'b-.', label='von Mises fit II')
ax.plot(x, sol_II[0]*Xvm_m, 'b-', label='weighted von Mises fit II')
ax.plot(x, Xu_m, 'g-.', label='Uniform fit II')
ax.plot(x, (1. - sol_II[0])*Xu_m, 'g-', label='weighted Uniform fit II')
ax.plot(x, X_m, 'r', label='Mixture fit II')
ax.legend()
fig_name = 'Fig_random_sample_mix_vM_Un_II'
fig.savefig(fig_name + '.eps')

# ---------------------------------------------------------------------- #
fig, ax = plt.subplots(1, 1, figsize=(9,3))
ax.hist( X_samples, bins=100, density=True, label='sample', color = 'skyblue' );
ax.plot(x, Xvm_b, color='blue', \
        label=r'single von Mises fit; $\mu$={}, $\kappa$={} '.format(round(m_bar,3), round(kappa0_[0],3)))
ax.plot(x, X_m, color='red', \
        label=r'mix fit 1vM1U; $\mu$ = {}, $\kappa$= {}, p= {} '.format(round(sol_II[2],3), round(sol_II[1],3), round(sol_II[0],3)))
ax.set_xlabel(r'$\theta$ (rad)', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=5)
ax.set_title('Random sample from mixture of von Mises and Uniform distributions')

