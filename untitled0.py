#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 18:14:58 2018

@author: df
"""

import numpy as np
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd

from dff_StatsTools import rs_mix_vonMises
#%%

# ---------------------------------------------------------------------- #
# some initial parameters to generate the random sample: 
# size of sample: 
N = 1000
# parameters for the von Mises member:
p1_ = 0.3                   # weight contribution of the 1st von Mises 
p3_ = 0.3                   # weight contribution of the 3rd von Mises
p2_ = 0.3 
#p2_ = 1. - p1_ - p3_        # weight contribution of the 2nd von Mises 
pu = 1. - p1_ - p2_ - p3_
kappa1_ = np.array((12.0))  # concentration for the 1st von Mises member 
kappa2_ = np.array((12.0))   # concentration for the 2nd von Mises member 
kappa3_ = np.array((12.0))   # concentration for the 3rd von Mises member 
loc1_ = -np.pi/3.0          # location for the 1st von Mises member 
loc2_ =  0.*np.pi/12.0      # location for the 2nd von Mises member 
loc3_ =  np.pi/3.0          # location for the 3rd von Mises member 
#loc_cs = np.array(( np.cos(loc_), np.sin(loc_) )) # cos and sin of location
#print('loc_cs = ',loc_cs)
kappas = np.array([ kappa1_, kappa2_, kappa3_ ]) 
locs = np.array([ loc1_, loc2_, loc3_ ])
pis = np.array([p1_, p2_, p3_, pu])
# parameter to scale the user-defined von Mises on the SEMI-CIRCLE 
ll = 2.0
# parameter to scale the stats von Mises on the SEMI-CIRCLE 
scal_ = 0.5

#%%
# ---------------------------------------------------------------------- #
# Generate the random sample:
# "Creating a mixture of probability distributions for sampling"
#   a question on stackoverflow: 
#   https://stackoverflow.com/questions/47759577/
#               creating-a-mixture-of-probability-distributions-for-sampling
sample_size = N
num_distr = 3+1
coefficients = pis
coefficients /= coefficients.sum() # in case these did not add up to 1
data = np.zeros((sample_size, num_distr))
data0 = np.zeros((sample_size, num_distr))
idx = 0
transfer_ = np.pi*scal_
for mu, kap in zip(locs, kappas):
    temp_ = stats.vonmises.rvs( kap, mu, scal_, size=N )
    data0[:, idx] = temp_
    if mu > 0.0:
        temp_u = temp_[(temp_ >= transfer_)]
        temp_l = temp_[(temp_ < transfer_)]
        temp_mod = np.concatenate((temp_u - 2.*transfer_, temp_l),axis=0)
    elif mu < 0.0:
        temp_l = temp_[(temp_ <= -transfer_)]
        temp_u = temp_[(temp_ > -transfer_)]
        temp_mod = np.concatenate((temp_u, temp_l + 2.*transfer_),axis=0)
    else:
        temp_mod = temp_
    data[:, idx] = temp_mod
    idx += 1

# if you want also the uniform distribution: 
u1_ = -np.pi/2
u2_ =  np.pi
data[:, idx] = stats.uniform.rvs( loc=u1_, scale=u2_, size=N )
data0[:, idx] = data[:, idx]

random_idx = np.random.choice( np.arange(num_distr), \
                              size=(sample_size,), p=coefficients )
X_samples0 = data[ np.arange(sample_size), random_idx ]
fig, ax = plt.subplots(1, 1, figsize=(9,3))
ax.hist( X_samples0, bins=100, density=True, label='sample-mod', color = 'skyblue' );
ax.set_title('Random sample from mixture of von Mises and Uniform distributions')
X_samples0_b = data0[ np.arange(sample_size), random_idx ]
ax.hist( X_samples0_b, bins=100, density=True, label='sample-original', color = 'orange', alpha=0.3 );
ax.legend()

X_samples = X_samples0
fig, ax = plt.subplots(1, 1, figsize=(9,3))
ax.hist( X_samples, bins=100, density=True, label='sample', color = 'skyblue' );
ax.set_title('Random sample from mixture of von Mises and Uniform distributions')
ax.legend()
