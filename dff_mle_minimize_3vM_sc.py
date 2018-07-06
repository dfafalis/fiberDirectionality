#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 19:51:17 2018
Author Dimitrios Fafalis 

This code generates a random sample from a mixture of THREE von Mises 
    on the SEMI-CIRCLE and ONE uniform distribution, 
    with weights p1, p2, p3 and (1-p1-p2-p3) 
    using the functions: "stats.vonmises.rvs" and "stats.uniform.rvs"
    using the "numpy.random.choice" function
    
This code makes use of "optimize.minimize" to minimize: 
    a. the negative log-likelihood function of the mixture of THREE von Mises
        distributions, with constraints 

@author: df
"""

import numpy as np
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
import dff_StatsTools as DFST
#from dff_StatsTools import rs_mix_vonMises, my_chisquare_GOF
#%%

# ---------------------------------------------------------------------- #
# some initial parameters to generate the random sample: 
# size of sample: 
N = 1000
# parameters for the von Mises member:
p1_ = 0.3                   # weight contribution of the 1st von Mises 
p3_ = 0.3                   # weight contribution of the 3rd von Mises 
#p2_ = 0.3                   # weight contribution of the 3rd von Mises 
#pu = 1. - p1_ - p2_ - p3_
p2_ = 1. - p1_ - p3_        # weight contribution of the 2nd von Mises 
kappa1_ = np.array((8.0))  # concentration for the 1st von Mises member 
kappa2_ = np.array((12.0))   # concentration for the 2nd von Mises member 
kappa3_ = np.array((8.0))   # concentration for the 3rd von Mises member 
loc1_ = -np.pi/3.0          # location for the 1st von Mises member 
loc2_ =  0.*np.pi/12.0      # location for the 2nd von Mises member 
loc3_ =  np.pi/3.0          # location for the 3rd von Mises member 
#loc_cs = np.array(( np.cos(loc_), np.sin(loc_) )) # cos and sin of location
#print('loc_cs = ',loc_cs)
kappas = np.array([ kappa1_, kappa2_, kappa3_ ]) 
locs = np.array([ loc1_, loc2_, loc3_])
pis = np.array([p1_, p2_, p3_])
#pis = np.array([p1_, p2_, p3_, pu])
# parameter to scale the user-defined von Mises on the SEMI-CIRCLE 
ll = 2.0
# parameter to scale the stats von Mises on the SEMI-CIRCLE 
scal_ = 0.5

#%%
#ka_ = 2.0
#lo_ = np.pi/2.0
#xs1_ = stats.vonmises.rvs( ka_, lo_, scal_, size=N )
#xs2_ = stats.vonmises.rvs( ka_, lo_, size=N )
#fig, ax = plt.subplots(1, 1, figsize=(9,3))
#ax.hist( xs2_, bins=100, density=True, label='circle', color = 'orange', alpha=0.6 );
#ax.hist( xs1_, bins=100, density=True, label='semi-circle', color = 'skyblue', alpha=0.8 );
#ax.legend()
#%%
#ka_ = 2.0
#lo_ = np.pi/2.0
#xs1_ = stats.vonmises_line.rvs( ka_, lo_, scal_, size=N )
#xs2_ = stats.vonmises_line.rvs( ka_, lo_, size=N )
#fig, ax = plt.subplots(1, 1, figsize=(9,3))
#ax.hist( xs1_, bins=100, density=True, label='semi-circle', color = 'skyblue', alpha=0.8 );
#ax.hist( xs2_, bins=100, density=True, label='circle', color = 'orange', alpha=0.5 );
#ax.legend()

# ---------------------------------------------------------------------- #
#%%
## ---------------------------------------------------------------------- #
## Generate the random sample:
## "Creating a mixture of probability distributions for sampling"
##   a question on stackoverflow: 
##   https://stackoverflow.com/questions/47759577/
##               creating-a-mixture-of-probability-distributions-for-sampling
#u1_ = -np.pi/2
#u2_ =  np.pi
#distributions = [
#        { "type": stats.vonmises.rvs, \
#                     "args": {"kappa":kappa1_, "loc":loc1_, "scale":scal_ }},
#        { "type": stats.vonmises.rvs, \
#                     "args": {"kappa":kappa2_, "loc":loc2_, "scale":scal_ }},
#        { "type": stats.vonmises.rvs, \
#                     "args": {"kappa":kappa3_, "loc":loc3_, "scale":scal_ }},
##        { "type": stats.uniform.rvs,  \
##                     "args": {"loc":u1_, "scale":u2_ } }
#]
#coefficients = np.array( [ p1_, p2_, p3_ ] ) # these are the weights 
##coefficients = np.array( [ p1_, p2_, p3_, pu ] ) # these are the weights 
##coefficients = np.array( [ p1_, p2_ ] ) # these are the weights 
#coefficients /= coefficients.sum() # in case these did not add up to 1
#sample_size = N
#
#transfer_ = np.pi*scal_
#num_distr = len(distributions)
#data = np.zeros((sample_size, num_distr))
#datab = np.zeros((sample_size, num_distr))
#for idx, distr in enumerate(distributions):
#    temp_ = distr["type"]( **distr["args"], size=(sample_size,))
#    datab[:, idx] = temp_
#    if locs[idx] > 0.0:
#    # if max(temp_) > transfer_:
#        temp_u = temp_[(temp_ >= transfer_)]
#        temp_l = temp_[(temp_ < transfer_)]
#        temp_mod = np.concatenate((temp_u - 2.*transfer_, temp_l),axis=0)
#    elif locs[idx] < 0.0:
#    # elif min(temp_) < -transfer_:
#        temp_l = temp_[(temp_ <= -transfer_)]
#        temp_u = temp_[(temp_ > -transfer_)]
#        temp_mod = np.concatenate((temp_u, temp_l + 2.*transfer_),axis=0)
#    else:
#        temp_mod = temp_
#    data[:, idx] = temp_mod
#    # data[:, idx] = distr["type"]( **distr["args"], size=(sample_size,))
#random_idx = np.random.choice( np.arange(num_distr), \
#                              size=(sample_size,), p=coefficients )
#X_samples0 = data[ np.arange(sample_size), random_idx ]
#fig, ax = plt.subplots(1, 1, figsize=(9,3))
#ax.hist( X_samples0, bins=100, density=True, label='sample-mod', color = 'skyblue' );
#ax.set_title('Random sample from mixture of von Mises and Uniform distributions')
#X_samples0_b = datab[ np.arange(sample_size), random_idx ]
#ax.hist( X_samples0_b, bins=100, density=True, label='sample-original', color = 'orange', alpha=0.3 );
#ax.legend()
## ---------------------------------------------------------------------- #
### the following is cropping the r.s. at the exceeding points: (not correct!) 
##xtemp_ = X_samples0.copy()
##X_samples_lower = xtemp_[xtemp_ > u1_]
##X_samples_upper = xtemp_[xtemp_ < -u1_]
##X_samples_cr = xtemp_[(xtemp_>u1_) & (xtemp_<-u1_)]
##ax.hist( X_samples_lower, bins=100, density=True, label='lower', color = 'orange', alpha=0.3 );
##ax.hist( X_samples_upper, bins=100, density=True, label='upper', color = 'green', alpha=0.3 );
##ax.hist( X_samples_cr, bins=100, density=True, label='upper-lower', color = 'm', alpha=0.3 );
##ax.legend()
##X_samples = X_samples_cr
### ---------------------------------------------------------------------- #
#
## CAUTION::: 
## rearrange the points, do not crop them!: 
## this has to be done for every individual distribution r.s. 
## (this means do it within the above for loop)
##xs_u = X_samples0[(X_samples0 >= lo_)]
##xs_l = X_samples0[(X_samples0 < lo_)]
##xs_mod = np.concatenate((xs_u - 2.*lo_, xs_l),axis=0)
##X_samples = xs_mod
#X_samples = X_samples0
#fig, ax = plt.subplots(1, 1, figsize=(9,3))
#ax.hist( X_samples, bins=100, density=True, label='sample', color = 'skyblue' );
#ax.set_title('Random sample from mixture of von Mises and Uniform distributions')
#ax.legend()
## ---------------------------------------------------------------------- #
#%%
X_samples = DFST.rs_mix_vonMises( kappas, locs, pis, sample_size=N )

fig, ax = plt.subplots(1, 1, figsize=(9,3))
ax.set_title('Random sample from mixture of von Mises distributions - Histogram')
(nX_samp, binsX_samp, patchesX_samp) = ax.hist( X_samples, bins=100, \
                            density=True, label='sample', color = 'skyblue' );
#ax.set_title('Random sample from mixture of von Mises and Uniform distributions')
ax.legend()


#%%
# ---------------------------------------------------------------------- #
def logLik_3vM( theta, *args ):
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
    #print('args0:',args[0])
    #print('args1:',args[1])
    x_ = np.array(args[0]).T
#    print(type(x_))
#    print('x_=', x_.shape)
    
    # the unknown parameters:
    p1_ = theta[0]
    kappa1_ = theta[1]
    m1_ = theta[2]
    p2_ = theta[3]
    kappa2_ = theta[4]
    m2_ = theta[5]
    p3_ = theta[6]
    kappa3_ = theta[7]
    m3_ = theta[8]
    
    # the 1st von Mises distribution on semi-circle:
    fvm1_ = stats.vonmises.pdf( x_, kappa1_, m1_, scal_ )
    # logvM1 = -np.sum( stats.vonmises.logpdf( x_, kappa1_, m1_, scal_ ) )
    
    # the 2nd von Mises distribution on semi-circle:
    fvm2_ = stats.vonmises.pdf( x_, kappa2_, m2_, scal_ )
    # logvM2 = -np.sum( stats.vonmises.logpdf( x_, kappa2_, m2_, scal_ ) )
        
    # the 3rd von Mises distribution on semi-circle:
    fvm3_ = stats.vonmises.pdf( x_, kappa3_, m3_, scal_ )
    
    # mixture distribution: 
    fm_ = p1_*fvm1_ + p2_*fvm2_ + p3_*fvm3_
    # logMix = logvM1 + logvM2 + logU - len(x_)*(np.log(p1_*p2_*pu_))
    
    logMix = -np.sum( np.log( fm_ ) )
    
    return logMix
# ---------------------------------------------------------------------- #

#%%
in_guess = [ p1_, kappa1_, loc1_, p2_, kappa2_, loc2_, p3_, kappa3_, loc3_ ]
#in_guess = [ p1_, kappa1_, loc1_, p2_, kappa2_, loc2_, 1.-p1_-p2_, kappa3_, loc3_ ]
# bound constraints for the variables: 
bnds = ((0., 1.), (0., 100.), (-np.pi/2, np.pi/2), \
        (0., 1.), (0., 100.), (-np.pi/2, np.pi/2), \
        (0., 1.), (0., 100.), (-np.pi/2, np.pi/2))
cons = ({'type': 'eq', 'fun': lambda x:  x[0] + x[3] + x[6] - 1.0})
results = optimize.minimize(logLik_3vM, in_guess, args=(X_samples,in_guess), \
                            method='SLSQP', bounds=bnds, constraints=cons, \
                            tol=1e-6, options={'maxiter': 100, 'disp': True})
print('METHOD II = ',results.x)
print('-----------------------------------------')
p1_mle, kappa1_mle, mu1_mle, p2_mle, kappa2_mle, mu2_mle, \
                             p3_mle, kappa3_mle, mu3_mle = results.x
print('p1, kappa1, mu1, p2, kappa2, mu2, p2, kappa2, mu2 = ',results.x)
res = results.x
#mu1_mle = -np.pi/6.0
#kappa1_mle = 5.0
data = np.array([[p1_mle, kappa1_mle, mu1_mle],
                 [p2_mle, kappa2_mle, mu2_mle],
                 [p3_mle, kappa3_mle, mu3_mle]])

# ---------------------------------------------------------------------- #
# Plot the fit in the same figure as the histogram:
members_list = ["1st von Mises", "2nd von Mises", "3rd von Mises"]
loc_mle = np.array([mu1_mle, mu2_mle, mu3_mle])
kap_mle = np.array([kappa1_mle, kappa2_mle, kappa3_mle])
p_mle = np.array([p1_mle, p2_mle, p3_mle])
dataFrame = pd.DataFrame({'Distribution': members_list, \
                          'Weight': p_mle.ravel(), \
                          'Concentration': kap_mle.ravel(), \
                          'Location': loc_mle.ravel()})
dataFrame = dataFrame[['Distribution', 'Weight', 'Concentration','Location']]
#dataFrame.set_index('Distribution')
print(dataFrame)

# plot in the same histogram the approximations:
#x_ = np.linspace( min(X_samples), max(X_samples), N )
x_ = np.sort(X_samples)
X_tot = np.zeros(len(x_),)
cXtot = np.zeros(len(x_),)
for mu, kap, pii in zip(loc_mle, kap_mle, p_mle):
    fX_temp = stats.vonmises( kap, mu, scal_ )
    # X_temp = pii*stats.vonmises.pdf( x_, kap, mu, scal_ )
    X_temp = pii*fX_temp.pdf( x_ )
    X_tot += X_temp
    ax.plot(x_, X_temp, linewidth=2, linestyle='--', \
            label=r'$\mu$ = {}, $\kappa$= {}, p= {} '.format(round(mu,3), round(kap,3), round(pii,3)))
    # cXtot += pii*stats.vonmises.cdf( x_, kap, mu, scal_ )
    cXtot += pii*fX_temp.cdf( x_ )
    
ax.plot(x_, X_tot, color='red', linewidth=2, linestyle=':', label='fit mixture')
ax.set_xlabel(r'$\theta$ (rad)', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
#ax.legend(loc=2)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=3)

# insert a breakpoint:
1/0
#%% New figure: 
# -------------------------------------------------------------------- #
fig, ax = plt.subplots(1, 3, figsize=(12,3))

# subplot 1: 
# -------------------- #
ax[1].set_title('Random sample from mixture of von Mises and Uniform distributions')
ax[0].hist( X_samples, bins=100, density=True, label='r.s. histogram', color = 'skyblue' );
ax[0].legend()

# subplot 2: 
# -------------------- #
# PLOT the "Uniform Probability Plot": 
aa = np.sort(X_samples)
b = aa
h0 = 1./(N + 1)
hN = N/(N + 1)
h = np.arange(h0, hN, h0)
#fig, ax = plt.subplots(1, 1, figsize=(9,3))
ax[1].plot(h, b, label='PPF r.s.')
ax[1].legend()

# subplot 3: 
# -------------------- #
# draw the "ECDF Plot" (or empirical distribution function ECDF): 
# a. get ECDF using the cumsum command (1st way): 
dx = np.diff(binsX_samp)
dd = np.cumsum(nX_samp*dx)
ax[2].plot(binsX_samp[0:-1], dd, 'b', label='ECDF (cumsum-1)')

# b. get ECDF using the cumsum command (2nd way): 
sX = np.cumsum(nX_samp)/sum(nX_samp)
ax[2].plot(binsX_samp[0:-1], sX, 'g', label='ECDF, (cumsum-2)')

# c. get ECDF using ECDF() function: 
xx2, ee2 = DFST.ECDF( X_samples )
ax[2].plot(xx2, ee2, 'c:', lw=2, label='ECDF (ECDF.py)')

ax[2].plot(x_, cXtot, 'r-', label='CDF fit')
ax[2].legend()
#ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
#          fancybox=True, shadow=True, ncol=3)

ax[0].plot(binsX_samp[0:-1], dd, label='CDF r.s.')
ax[0].legend()


# -------------------------------------------------------------------- #
# create a separate plot for ECDF and CDF: 
fig, ax = plt.subplots(1, 1, figsize=(4,3))
ax.set_title('CDF of mixture of von Mises distributions')
ax.set_ylabel('Cumulative Probability')
ax.set_xlabel(r'$\theta$ (rad)', fontsize=12)
ax.plot(xx2, ee2, 'b-', label='ECDF')
ax.plot(x_, cXtot, 'r-.', label='CDF model')
ax.legend()

#%% 
# -------------------------------------------------------------------- #
# compute Watson and Kuiper statistics with ORIGINAL random data, 
#   NOT equally-spaced: 
#   so the CDF is computed over the ordered r.s. 

U2, Us, uc, pu2, pus = DFST.watson( cXtot, alphal=2 )
print('1st DFst.watson =',U2, Us, uc, pu2, pus)

Vn, pVn = DFST.Kuiper_GOF( cXtot )
print('1st Kuiper:', Vn, pVn)

# make the P-P plot: 
cXtot = np.zeros(len(xx2),)
for mu, kap, pii in zip(loc_mle, kap_mle, p_mle):
    fX_temp = stats.vonmises( kap, mu, scal_ )
    cXtot += pii*fX_temp.cdf( xx2 )
    
DFST.PP_GOF( cXtot, ee2 )

# compute the R2 coefficient: 
R2 = DFST.myR2( cXtot, ee2 )
print('R2:', R2)

Dn = DFST.my_KS_GOF_mvM( X_samples, dataFrame, alpha=0.05 )

# compute the K-S statistic: 
d00 = np.array([1.63, 1.36, 1.22])/np.sqrt(len(ee2))
Dm = abs(cXtot - ee2)
D = max(Dm)
sqnD = np.sqrt(len(ee2))*D
pval = 1 - DFST.KS_CDF( sqnD )
print('3rd K-S =',D, sqnD, pval, d00)

# the following yields wrong conclusion: 
U2, Us, uc, pu2, pus = DFST.watson( cXtot, alphal=2)
print('3rd DFst.watson =', U2, Us, uc, pu2, pus)

# insert a breakpoint:
1/0
#%%

# the chi-square GOF test: 
c2v = DFST.my_chisquare_GOF( X_samples, dataFrame, alpha=0.05 )

# the K-S GOF test: 
Dks = DFST.my_KS_GOF_mvM( X_samples, dataFrame, alpha=0.05 )


x, cfX_obs = DFST.ECDF( X_samples )

dx = np.diff(x)
R2 = DFST.myR2( np.cumsum(X_tot[0:-1]*dx), cfX_obs[0:-1] )
print('R2:', R2)
    
DFST.PP_GOF( np.cumsum(X_tot[0:-1]*dx), cfX_obs[0:-1] )

#U2, Us, uc, pu2, pus = DFST.watson( np.cumsum(X_tot[0:-1]*dx), alphal=2)
U2, Us, uc, pu2, pus = DFST.watson( cXtot, alphal=2)
print('1st DFst.watson =',U2, Us, uc, pu2, pus)

fig, ax = plt.subplots(1,1,figsize=(4,3))
ax.plot(x, cfX_obs, 'b', lw=3, alpha=0.6, label='CDF data')
ax.plot(x[0:-1], np.cumsum(X_tot[0:-1]*dx), 'r--', lw=2, alpha=0.6, label='CDF model')
#ax.plot(x, cXtot, 'g', lw=2, alpha=0.6, label='CDF fit +=')
ax.set_title('CDF plot')
ax.set_xlabel(r'$\theta$ (rads)', fontsize=12)
ax.set_ylabel('Cumulative distribution', fontsize=12)
ax.legend()

#%% For the Watson's test: 
# plot in the same histogram the approximations:
#aa = np.linspace(-np.pi*scal_, np.pi*scal_, 100)
aa = np.sort(X_samples)
fX_t = np.zeros(len(aa),)
cX_t = np.zeros(len(aa),)
for mu, kap, pii in zip(loc_mle, kap_mle, p_mle):
    cX_t += pii*stats.vonmises.cdf( aa, kap, mu, scal_ )
    fX_t += pii*stats.vonmises.pdf( aa, kap, mu, scal_ )

fig, ax = plt.subplots(1,1,figsize=(4,3))
ax.set_title('CDF of mix von Mises, sorted')
ax.set_xlabel(r'$\theta$ (rads)', fontsize=12)
ax.set_ylabel('Cumulative distribution', fontsize=12)
ax.plot(aa, cX_t, 'r', label='CDF fit')
#ax.plot(x, cfX_obs, 'b', lw=3, alpha=0.6, label='CDF data')
dx = np.diff(aa)
ax.plot(aa[0:-1], np.cumsum(fX_t[0:-1]*dx), 'g--', lw=2, alpha=0.6, label='CDF model')
ax.legend()

U2, Us, uc, pu2, pus = DFST.watson( cX_t, alphal=2)
print('2nd DFst.watson =',U2, Us, uc, pu2, pus)
U2, Us, uc, pu2, pus = DFST.watson( np.cumsum(fX_t[0:-1]*dx), alphal=2)
print('3rd DFst.watson =',U2, Us, uc, pu2, pus)

# insert a breakpoint:
1/0
# ---------------------------------------------------------------------- #
#%%
# random sample from the fitted mixture distribution:
X_fit = rs_mix_vonMises( kap_mle, loc_mle, p_mle, sample_size=N )
chi2, pvalue = stats.chisquare( X_samples, f_exp=X_fit )
print('chi2= ',chi2, 'p= ',pvalue)

(n, bins, patches) = plt.hist(X_fit, bins=10, label='hst')

fig, ax = plt.subplots(1, 1, figsize=(9,6))
ax.plot(X_samples,label='observed')
ax.plot(X_fit,label='expected')
ax.legend(loc=2)

fig, ax = plt.subplots(1, 1, figsize=(9,6))
(n1, bins1, patches1) = ax.hist( X_samples, bins=50, density=True, label='observed', color = 'skyblue', alpha=0.3 )
(n2, bins2, patches2) = ax.hist( X_fit, bins=50, density=True, label='expected', color = 'orange', alpha=0.3 );
ax.legend(loc=2)

chi_squared_ = sum(((n1 - n2)**2)/n2)
stats.chi2.cdf( x=chi_squared_, df=50-1-6 )

chi_squared_stat = np.sum(((X_samples - X_fit)**2)/X_fit)
print('chichi_squared_stat23= ',chi_squared_stat)

# Find the critical value for 95% confidence: 
crit = stats.chi2.ppf( q = 0.95, df = 8 )
print("Critical value")
print(crit)

# Find the p-value: 
p_value = 1 - stats.chi2.cdf( x=chi_squared_stat, df=8 )
print("P value")
print(p_value)

#stats.kstest(X_fit, 'norm')
#stats.kstest(X_fit, 'uniform')
#stats.kstest(X_fit, 'vonmises', )

#%%
np.random.seed(987654321)
x03 = stats.t.rvs(3,size=100)
fig, ax = plt.subplots(figsize=(9,6))
ax.hist(x03, bins=100, density=True, label='3');
np.random.seed(987654321)
x100 = stats.t.rvs(100,size=100)
ax.hist(x100, bins=100, density=True, label='100');
ax.legend(loc=2)

stats.kstest(x03,'norm')
stats.kstest(x100,'norm')

# for a normal r.s.: 
np.random.seed(987654321)
xn = stats.norm.rvs(5, 0.2, size=1000)
fig, ax = plt.subplots(figsize=(9,6))
(n_xn, bins_xn, patches) = ax.hist(xn, bins=1000, density=True, label='normal');
print('K-S test - normal: ', stats.kstest(n_xn,'norm', alternative = 'greater'))
print('K-S test - uniform: ', stats.kstest(n_xn,'uniform', alternative = 'greater'))
print('chichi_squared test: ', stats.chisquare(n_xn))

# for a uniform r.s.: 
np.random.seed(987654321)
xu = stats.uniform.rvs( loc=0, scale=1, size=1000)
fig, ax = plt.subplots(figsize=(9,6))
(n_xu, bins_xu, patches) = ax.hist(xu, bins=100, density=True, label='uniform')
print('K-S test - normal: ', stats.kstest(n_xu,'norm', alternative = 'greater'))
print('K-S test - uniform: ', stats.kstest( n_xu, 'uniform', alternative = 'greater'))
print('chichi_squared test: ', stats.chisquare(n_xu))

chi2, pvalue = stats.chisquare(xu)
print('chi2= ;',chi2, 'p= ',pvalue)

chi2, pvalue = stats.chisquare(xn)
print('chi2 =',chi2, 'p =',pvalue)

