#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:33:10 2018

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

## ---------------------------------------------------------------------- #
# generate random variables from a mixture of von Mises and \
# uniform distributions:
N = 1000
# for the von Mises member:
p = 0.8 # weight contribution of von Mises 
kappa_ = np.array((2.0)) # concentration for von Mises member 
loc_ = np.pi/4 # location for von Mises member 
loc_cs = np.array(( np.cos(loc_), np.sin(loc_) ))
print(loc_cs)

# X = stats.vonmises.rvs(kappa, loc, size=N)
x = np.linspace(stats.vonmises.ppf(0.001, kappa_, loc_ ), \
                stats.vonmises.ppf(0.999, kappa_, loc_ ), N)
Xvm_ = stats.vonmises.pdf(x, kappa_, loc_ )
plt.plot(x, Xvm_, 'r-', lw=5, alpha=0.6, label='vonmises pdf')
plt.plot(x, p*Xvm_, 'b-', lw=5, alpha=0.6, label='vonmises pdf')
plt.plot(x, (1-p)*Xvm_, 'g-', lw=5, alpha=0.6, label='vonmises pdf')

# for the uniform distribution:
#xu = np.linspace(stats.uniform.ppf(0.01, loc=-np.pi, scale=2*np.pi ), \
#                stats.uniform.ppf(0.99, loc=-np.pi, scale=2*np.pi ), N)
#Xu_ = stats.uniform.pdf(xu, loc=-np.pi, scale=2*np.pi )
xu = np.linspace(stats.uniform.ppf( 0.01, loc=min(x), scale=(max(x)-min(x)) ), \
                 stats.uniform.ppf( 0.99, loc=min(x), scale=(max(x)-min(x)) ), N)
Xu_ = stats.uniform.pdf(xu, loc=min(x), scale=(max(x)-min(x)) )
plt.plot(xu, Xu_, 'r-', lw=5, alpha=0.6, label='vonmises pdf')
plt.plot(xu, p*Xu_, 'b-', lw=5, alpha=0.6, label='vonmises pdf')
plt.plot(xu, (1-p)*Xu_, 'g-', lw=5, alpha=0.6, label='vonmises pdf')

plt.plot(xu, Xu_, 'r-', lw=5, alpha=0.6, label='vonmises pdf')
plt.plot(xu, p*Xu_, 'b-', lw=5, alpha=0.6, label='vonmises pdf')
plt.plot(xu, (1-p)*Xu_, 'g-', lw=5, alpha=0.6, label='vonmises pdf')


## ---------------------------------------------------------------------- #
Xvm_ = stats.vonmises.pdf(kappa_, loc_)
Xvm = stats.vonmises(kappa_, loc_)
Xvm_sam = Xvm.rvs(size=N)
dfst.plot_dist_samples(Xvm, Xvm_sam, title='von Mises', ax=None)
# uniform distribution:
# A uniform continuous random variable
# This distribution is constant between loc and loc + scale
Xu = stats.uniform( loc=-np.pi, scale=2*np.pi )
Xu_sam = (0.5/np.pi)*stats.uniform.rvs( loc=-np.pi, scale=2*np.pi, size=N )
dfst.plot_dist_samples( Xu, Xu_sam, title='Uniform', ax=None )

Xu_smp = (1-p)*Xu_sam
Xvm_smp = p*Xvm_sam
Xmix_sum = Xvm_sam + Xu_sam
Xmix_smp = p*Xvm_sam + (1-p)*Xu_sam

fig, ax = plt.subplots(1, 2, figsize=(6,3))
ax[0].plot(Xu_sam, 'blue')
ax[1].plot(Xu_smp, 'orange')

fig, ax = plt.subplots(1, 2, figsize=(6,3))
ax[0].plot(Xvm_sam, 'blue')
ax[1].plot(Xvm_smp, 'orange')


fig, ax = plt.subplots(1, 4, figsize=(16, 3))
ax[0].hist(Xvm_sam, label='von Mises', normed=1, bins=75, color = 'skyblue');
ax[1].hist(Xu_sam, label='Uniform', normed=1, bins=75, color = 'green');
ax[2].hist(Xvm_sam, label='von Mises', normed=1, bins=75, color = 'skyblue');
ax[2].hist(Xu_sam, label='Uniform', normed=1, bins=75, color = 'green');
ax[2].hist(Xmix_sum, label='mixture', normed=1, bins=75, color = 'orange');
ax[3].hist(Xvm_smp, label='von Mises', normed=1, bins=75, color = 'skyblue');
ax[3].hist(Xu_smp, label='Uniform', normed=1, bins=75, color = 'green');
ax[3].hist(Xmix_smp, label='mixture', normed=1, bins=75, color = 'orange');
for ax in ax:
    ax.legend(loc='best')
## ---------------------------------------------------------------------- #

distributions = [
    {"type": np.random.normal, "kwargs": {"loc": -3, "scale": 2}},
    {"type": np.random.uniform, "kwargs": {"low": 4, "high": 6}},
    {"type": np.random.normal, "kwargs": {"loc": 2, "scale": 1}},
]
coefficients = np.array([0.5, 0.2, 0.3])
coefficients /= coefficients.sum()      # in case these did not add up to 1
sample_size = 100000

num_distr = len(distributions)
data = np.zeros((sample_size, num_distr))
for idx, distr in enumerate(distributions):
    data[:, idx] = distr["type"](size=(sample_size,), **distr["kwargs"])
random_idx = np.random.choice(np.arange(num_distr), size=(sample_size,), p=coefficients)
sample = data[np.arange(sample_size), random_idx]
plt.hist(sample, bins=100, density=True)
plt.show()

## ---------------------------------------------------------------------- #

distributions = [
    {"type": stats.vonmises.rvs, "args": {"kappa":kappa_, "loc":loc_ }},
    {"type": stats.uniform.rvs, "args": {"loc":-np.pi, "scale":2*np.pi}},
]
coefficients = np.array([0.8, 0.2])
coefficients /= coefficients.sum()      # in case these did not add up to 1
sample_size = N

num_distr = len(distributions)
data = np.zeros((sample_size, num_distr))
for idx, distr in enumerate(distributions):
    data[:, idx] = distr["type"]( **distr["args"], size=(sample_size,))
random_idx = np.random.choice(np.arange(num_distr), size=(sample_size,), p=coefficients)
sample = data[np.arange(sample_size), random_idx]
plt.hist(sample, bins=100, density=True );
plt.title('Random sample from a mixture of von Mises and Uniform distriburtions')
