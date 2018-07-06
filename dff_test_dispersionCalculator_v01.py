# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:53:34 2018

@author: DF
"""

#import csv
import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from scipy import stats
import pandas as pd

# import packages specific to our work: 
import dff_dispersionCalculator as dC
import dff_mle_minimize_mvMU as DFmle
import dff_StatsTools as DFST

#import dff_mle_fsolve_mvMU as DFfsolve

import time
import os, os.path

#from PIL import Image
# import glob

# Define the path where the data from the FFT (the .csv file) is located: 
# myPath = '/Users/df/0_myDATA/testSamples/'
myPath = '/Users/df/Documents/myGits/fiberDirectionality/testSamples/'

#YNtest = 'No'
YNtest = 'myMix'

if YNtest == 'myMix':
    print('testing my MLE ...')
    
# ------------------------------------------------------------------------- #
#    im_path = myPath + 'MAX_20X_Airyscan_6.jpg'
##    csv_path = myPath + "MAX_20X_Airyscan_6.csv"
#    csv_path = myPath + "MAX_20X_Airyscan_6_90.csv"
    
#    im_path = myPath + 'MAX_20180223_S2_20X_2c_CenterCropped3.png'
#    csv_path = myPath + "MAX_20180223_S2_20X_2c_CenterCropped3_FFT.csv"
    
    im_path = myPath + 'Clipboard.png'
    csv_path = myPath + "Dir_hist_FFT.csv"
#    csv_path = myPath + "Dir_hist_FFT_180.csv"
    
#    im_path = myPath + 'InCos.png'
#    csv_path = myPath + "InCos0pi.csv"
    
    temp_path, temp_file = os.path.split(im_path)
    print(temp_path)
    print(temp_file)
# ------------------------------------------------------------------------- #
    
    # read the data from the csv file: 
    angles, values, mydat = dC.imageCVS2Data( csv_path )
    c, s, r, rb, al0, al0d = dC.circ_measures(angles, values)
    print(c, s, r, rb, al0, al0d)
    # convert the angles from degrees to radians:
    angles = angles# - 90
    r_X = np.radians( angles )
    X_samples = r_X
    # compute the cosines and sines of the radiant angles r_X:
    # X is the input to the fit function of spherecluster module.
    # X = dC.getCosSin( r_X )
    
    # normalize the light intensity data (FFT): 
    n_X = dC.normalizeIntensity( angles, values, YNplot='Yes' )
    Int = n_X[:,1]
    
    # make points of angles based on the light intensity: 
    p_X = dC.makePoints( n_X, YNplot='Yes' )
    
    # mix3 = DFmle.logLik_3vM( theta, *args )
    
    #model_test = '1vM'
    #model_test = '2vM'
    #model_test = '3vM'
    #model_test = '2vM1U'
    model_test = '1vM1U'
    # CAUTION: sensitive to the initial guess! so decide first which values 
    #           to use based on the histogram of the original data.
    #%% for a 3-vM model: 
    #model_test = '3vM'
    if model_test == '3vM':
        # parameters for the von Mises member:
        p1_ = 0.4                   # weight contribution of the 1st von Mises 
        p3_ = 0.3                   # weight contribution of the 3rd von Mises 
        #p2_ = 0.2                   # weight contribution of the 3rd von Mises 
        p2_ = 1. - p1_ - p3_        # weight contribution of the 2nd von Mises 
        kappa1_ = np.array((12.0))  # concentration for the 1st von Mises member 
        kappa2_ = np.array((5.0))   # concentration for the 2nd von Mises member 
        kappa3_ = np.array((12.0))   # concentration for the 3rd von Mises member 
        loc1_ = -np.pi/3.0          # location for the 1st von Mises member 
        loc2_ = -0.*np.pi           # location for the 2nd von Mises member 
        loc3_ =  np.pi/3.0          # location for the 3rd von Mises member 
        
        # ------------------------------------- # 
        # if you solve with minimize: 
        in_guess = [ p1_, kappa1_, loc1_, p2_, kappa2_, loc2_, p3_, kappa3_, loc3_ ]
        # bound constraints for the variables: 
        lim_l = -np.pi/2.
        lim_u =  np.pi/2.
        bnds = ((0., 1.), (0., 100.), (lim_l, lim_u), \
                (0., 1.), (0., 100.), (lim_l, lim_u), \
                (0., 1.), (0., 100.), (lim_l, lim_u))
        cons = ({'type': 'eq', 'fun': lambda x:  x[0] + x[3] + x[6] - 1.0})
        results = optimize.minimize(DFmle.logLik_3vM, in_guess, args=(r_X,Int), \
                                    method='SLSQP', bounds=bnds, constraints=cons, \
                                    tol=1e-6, options={'maxiter': 100, 'disp': True})
        print('METHOD II = ',results.x)
        print('-----------------------------------------')
        p1_mle, kappa1_mle, mu1_mle, p2_mle, kappa2_mle, mu2_mle, \
                                                p3_mle, kappa3_mle, mu3_mle = results.x
        print('p1, kappa1, mu1, p2, kappa2, mu2, p2, kappa2, mu2 = ',results.x)
        res = results.x
        # ------------------------------------- #   
        # data = np.array([[p1_mle, kappa1_mle, mu1_mle],
        #                  [p2_mle, kappa2_mle, mu2_mle],
        #                  [p3_mle, kappa3_mle, mu3_mle]])
        
        # ---------------------------------------------------------------------- #
        # Plot the fit in the same figure as the histogram:
        members_list = ["1st von Mises", "2nd von Mises", "3rd von Mises"]
        loc_mle = np.array([mu1_mle, mu2_mle, mu3_mle])
        kap_mle = np.array([kappa1_mle, kappa2_mle, kappa3_mle])
        p_mle = np.array([p1_mle, p2_mle, p3_mle])

    #%% for a 2-vM model: 
    #model_test = '2vM'
    if model_test == '2vM':
        # parameters for the von Mises member:
        p1_ = 0.4             # weight contribution of the 1st von Mises 
        p2_ = 1. - p1_        # weight contribution of the 2nd von Mises 
        kappa1_ = np.array((5.0))  # concentration for the 1st von Mises member 
        kappa2_ = np.array((12.0))   # concentration for the 2nd von Mises member 
        loc1_ = -np.pi/9.0          # location for the 1st von Mises member 
        loc2_ = 0.*np.pi/9.0      # location for the 2nd von Mises member 
        
        # ------------------------------------- # 
        # if you solve with minimize: 
        in_guess = [ p1_, kappa1_, loc1_, p2_, kappa2_, loc2_ ]
        # bound constraints for the variables: 
        lim_l = -np.pi/2.
        lim_u =  np.pi/2.
        bnds = ((0., 1.), (0., 100.), (lim_l, lim_u), \
                (0., 1.), (0., 100.), (lim_l, lim_u))
        cons = ({'type': 'eq', 'fun': lambda x:  x[0] + x[3] - 1.0})
        results = optimize.minimize(DFmle.logLik_2vM, in_guess, args=(r_X,Int), \
                                    method='SLSQP', bounds=bnds, constraints=cons, \
                                    tol=1e-6, options={'maxiter': 100, 'disp': True})
        print('METHOD II = ',results.x)
        print('-----------------------------------------')
        p1_mle, kappa1_mle, mu1_mle, p2_mle, kappa2_mle, mu2_mle = results.x
        print('p1, kappa1, mu1, p2, kappa2, mu2 = ',results.x)
        res = results.x
        # ------------------------------------------------ #   
        # data = np.array([[p1_mle, kappa1_mle, mu1_mle],
        #                  [p2_mle, kappa2_mle, mu2_mle]])
        
        # ---------------------------------------------------------------------- #
        # Plot the fit in the same figure as the histogram:
        members_list = ["1st von Mises", "2nd von Mises"]
        loc_mle = np.array([mu1_mle, mu2_mle])
        kap_mle = np.array([kappa1_mle, kappa2_mle])
        p_mle = np.array([p1_mle, p2_mle])
    
    #%% for a 2vM1U model: 
    #model_test = '2vM1U'
    if model_test == '2vM1U':
        # parameters for the von Mises member:
        p1_ = 0.4                   # weight contribution of the 1st von Mises 
        p2_ = 0.4                   # weight contribution of the 2nd von Mises 
        pu_ = 1. - p1_ - p2_        # weight contribution of Uniform distribut 
        kappa1_ = np.array((12.0))  # concentration for the 1st von Mises member 
        kappa2_ = np.array((5.0))   # concentration for the 2nd von Mises member 
        loc1_ = -np.pi/9.0          # location for the 1st von Mises member 
        loc2_ = -0.*np.pi/20.0      # location for the 2nd von Mises member 
        
        # ------------------------------------- # 
        # if you solve with minimize: 
        in_guess = [ p1_, kappa1_, loc1_, p2_, kappa2_, loc2_, pu_ ]
        # bound constraints for the variables: 
        lim_l = -np.pi/2.
        lim_u =  np.pi/2.
        bnds = ((0., 1.), (0., 100.), (lim_l, lim_u), \
                (0., 1.), (0., 100.), (lim_l, lim_u), \
                (0., 1.))
        cons = ({'type': 'eq', 'fun': lambda x:  x[0] + x[3] + x[6] - 1.0})
        results = optimize.minimize(DFmle.logLik_2vM1U, in_guess, args=(r_X,Int), \
                                    method='SLSQP', bounds=bnds, constraints=cons, \
                                    tol=1e-6, options={'maxiter': 100, 'disp': True})
        print('METHOD II = ',results.x)
        print('-----------------------------------------')
        p1_mle, kappa1_mle, mu1_mle, p2_mle, kappa2_mle, mu2_mle, \
                                                         pu_mle = results.x
        print('p1, kappa1, mu1, p2, kappa2, mu2, pu = ',results.x)
        res = results.x
        # ------------------------------------- #   
        # data = np.array([[p1_mle, kappa1_mle, mu1_mle],
        #                  [p2_mle, kappa2_mle, mu2_mle],
        #                  [pu_mle, 0.0,        0.0]])
        
        # ---------------------------------------------------------------------- #
        # Plot the fit in the same figure as the histogram:
        members_list = ["1st von Mises", "2nd von Mises", "Uniform"]
        loc_mle = np.array([mu1_mle, mu2_mle, 0.0])
        kap_mle = np.array([kappa1_mle, kappa2_mle, 1e-3])
        p_mle = np.array([p1_mle, p2_mle, pu_mle])
    
    #%% for a 1-vM model: 
    #model_test = '1vM'
    if model_test == '1vM':
        # parameters for the von Mises member:
        p1_ = 1.0             # weight contribution of the 1st von Mises 
        kappa1_ = np.array((6.0))  # concentration for the 1st von Mises member 
        loc1_ = -np.pi/3.0           # location for the 1st von Mises member 
        
        # ------------------------------------- # 
        # if you solve with minimize: 
        in_guess = [ kappa1_, loc1_ ]
        # bound constraints for the variables: 
        lim_l = -np.pi/2.
        lim_u =  np.pi/2.
        bnds = ((0., 100.), (lim_l, lim_u))
        #cons = ({'type': 'eq', 'fun': lambda x:  x[0] + x[3] - 1.0})
        results = optimize.minimize(DFmle.logLik_1vM, in_guess, args=(r_X,Int), \
                                    method='SLSQP', bounds=bnds, \
                                    tol=1e-6, options={'maxiter': 100, 'disp': True})
        print('METHOD II = ',results.x)
        print('-----------------------------------------')
        kappa1_mle, mu1_mle = results.x
        print('kappa1, mu1 = ',results.x)
        res = results.x
        # ------------------------------------------------ #   
        # data = np.array([[p1_mle, kappa1_mle, mu1_mle]])
        
        # ---------------------------------------------------------------------- #
        # Plot the fit in the same figure as the histogram:
        members_list = ["1st von Mises"]
        loc_mle = np.array([mu1_mle])
        kap_mle = np.array([kappa1_mle])
        p_mle = np.array([p1_])
    
    #%% for a 1vM1U model: 
    #model_test = '1vM1U'
    if model_test == '1vM1U':
        # parameters for the von Mises member:
        p1_ = 0.7                   # weight contribution of the 1st von Mises 
        pu_ = 1. - p1_              # weight contribution of Uniform distribut 
        kappa1_ = np.array((12.0))  # concentration for the 1st von Mises member 
        loc1_ = -np.pi/9.0          # location for the 1st von Mises member 
        
        # ------------------------------------- # 
        # if you solve with minimize: 
        in_guess = [ p1_, kappa1_, loc1_, pu_ ]
        # bound constraints for the variables: 
        lim_l = -np.pi/2.
        lim_u =  np.pi/2.
        bnds = ((0., 1.), (0., 100.), (lim_l, lim_u), \
                (0., 1.))
        cons = ({'type': 'eq', 'fun': lambda x:  x[0] + x[3] - 1.0})
        results = optimize.minimize(DFmle.logLik_1vM1U, in_guess, args=(r_X,Int), \
                                    method='SLSQP', bounds=bnds, constraints=cons, \
                                    tol=1e-6, options={'maxiter': 100, 'disp': True})
        print('METHOD II = ',results.x)
        print('-----------------------------------------')
        p1_mle, kappa1_mle, mu1_mle, pu_mle = results.x
        print('p1, kappa1, mu1, pu = ',results.x)
        res = results.x
        # ------------------------------------- #   
        # data = np.array([[p1_mle, kappa1_mle, mu1_mle],
        #                  [pu_mle, 0.0,        0.0]])
        
        # ---------------------------------------------------------------------- #
        # Plot the fit in the same figure as the histogram:
        members_list = ["1st von Mises", "Uniform"]
        loc_mle = np.array([mu1_mle, 0.0])
        kap_mle = np.array([kappa1_mle, 1e-3])
        p_mle = np.array([p1_mle, pu_mle])
    
    #%% collect the data into a dataFrame of Pandas: 
    #loc_mle_d = np.degrees(loc_mle + np.pi)
    loc_mle_d = np.degrees(loc_mle)
    dataFrame = pd.DataFrame({'Distribution': members_list, \
                              'Weight': p_mle.ravel(), \
                              'Concentration': kap_mle.ravel(), \
                              'Location': loc_mle.ravel(), \
                              'Location (deg)': loc_mle_d.ravel()})
    dataFrame = dataFrame[['Distribution', 'Weight', \
                           'Concentration','Location', 'Location (deg)']]
    #dataFrame.set_index('Distribution')
    print(dataFrame)
    
    #%% ------------------------------------------------------------------- # 
    scal_ = 0.5
    fig, ax = plt.subplots(1, 1, figsize=(9,4))
#    ax.set_title('Probability Density Functions of members and mixture')
    ax.set_title('Probability Density Functions of Mixture of von Mises')
    ax.plot(angles, Int, 'b-', label='Original data')
    # plot in the same histogram the approximations:
    x_ = np.linspace( min(X_samples), max(X_samples), len(r_X) )
    r_ = np.degrees(x_)
    x_ax = r_
    X_tot = np.zeros(len(x_),)
    cXtot = np.zeros(len(x_),)
    jj = 0
    for mu, kap, pii in zip(loc_mle, kap_mle, p_mle):
        jj += 1
        fX_temp = stats.vonmises( kap, mu, scal_ )
        # X_temp = pii*stats.vonmises.pdf( x_, kap, mu, scal_ )
        X_temp = pii*fX_temp.pdf ( x_ )
        X_tot += X_temp
        ax.plot(x_ax, X_temp, linewidth=2, linestyle='--', \
                label='von Mises member {} '.format(jj))
                #label=r'$\mu$ = {}, $\kappa$= {}, p= {} '.format(round(mu,3), round(kap,3), round(pii,3)))
        # this is wrong!!!: 
        # cXtot += pii*stats.vonmises.cdf( x_, kap, mu, scal_ )
        cXtot += pii*fX_temp.cdf( x_ )
        
    ax.plot(x_ax, X_tot, color='red', linewidth=2, linestyle='-', label='Mixture fit')
    ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
    ax.set_ylabel(r'$f(\theta)$', fontsize=12)
    ax.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)
    ax.legend(loc=1)
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
    #          fancybox=True, shadow=True, ncol=3)
    
    
    #%% ------------------------------------------------------------------- #
    # create a separate plot for ECDF and CDF: 
    # CAUTION !!!! 
    # this gives the WRONG ECDGF because the X_samples on which it is based
    #   is not a r.s.; it is the equally-spaced angles over which the 
    #   light intensity was measured with the FFT.
    # c. get ECDF using ECDF() function: 
    xx2, ee2 = DFST.ECDF( X_samples )
    
    fig, ax = plt.subplots(1, 1, figsize=(4,3))
    ax.set_title('CDF of mixture of von Mises distributions (wrong)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_xlabel(r'$\theta$ (rad)', fontsize=12)
    ax.plot(xx2, ee2, 'b-', label='ECDF')
    ax.plot(x_, cXtot-min(cXtot), 'r-.', label='CDF model')
    ax.legend()
    
    #%% GOF with intensity data: 
    
    Dks = DFST.my_KS_GOF_mvM_I( n_X, dataFrame, alpha=0.05 )
    
    x, cfX_obs = DFST.ECDF_Intensity( angles, values )
    
    dx = abs(x[0] - x[1])
    fig, ax = plt.subplots(1,1,figsize=(4,3))
    ax.set_title('CDF plot')
    ax.plot(x, cfX_obs, 'b', lw=3, alpha=0.6, label='ECDF, ecdf()')
    ax.plot(x, np.cumsum(X_tot*dx), 'r--', lw=2, alpha=0.6, label='CDF model')
    #ax.plot(x, cXtot, 'g', lw=2, alpha=0.6, label='CDF fit +=')
    ax.set_xlabel(r'$\theta$ (rads)', fontsize=12)
    ax.set_ylabel('Cumulative distribution', fontsize=12)
    ax.legend()
    
    R2 = DFST.myR2( np.cumsum(X_tot*dx), cfX_obs )
    print('R2:', R2)
    
    DFST.PP_GOF( np.cumsum(X_tot*dx), cfX_obs )
    
    # this returns wrong conclusion: 
    U2, Us, uc, pu2, pus = DFST.watson_GOF( np.cumsum(X_tot*dx), alphal=2)
    print('1st DFst.watson =',U2, Us, uc, pu2, pus)
    
    #%% ------------------------------------------------------------------- #
    # based on populated points for every angle based on its intensity: 
    # CAUTION !!!
    #   THIS IS THE CORRECT APPROACH !!!
    aa = np.sort( p_X )
    fX_t = np.zeros(len(aa),)
    cX_t = np.zeros(len(aa),)
    for mu, kap, pii in zip(loc_mle, kap_mle, p_mle):
        temp = stats.vonmises( kap, mu, scal_ )
        fX_t += pii*temp.pdf( aa )
        cX_t += pii*temp.cdf( aa )
        #fX_t += pii*stats.vonmises.pdf( aa, kap, mu, scal_ )
        #cX_t += pii*stats.vonmises.cdf( aa, kap, mu, scal_ )
    
    x1, cfX_obs = DFST.ECDF( p_X )
    
    dx = np.diff(aa)
    cX_b = np.ones(len(aa),)
    cX_b[0:-1] = np.cumsum(fX_t[0:-1]*dx)
    
    if max(cX_t) > 1:
        cX_c = cX_t - (max(cX_t) - 1.)
    else:
        cX_c = cX_t
    
    fig, ax = plt.subplots(1,1,figsize=(4,3))
    ax.set_title('CDF of mixture of von Mises distributions (p_X)')
    ax.set_xlabel(r'$\theta$ (rads)', fontsize=12)
    ax.set_ylabel('Cumulative distribution', fontsize=12)
    ax.plot( aa, cX_t, 'r', label='CDF fit (+=)')
    ax.plot( aa, cX_t-min(cX_t), 'b--', label='CDF fit (+=)b')
    ax.plot( aa, cX_c, 'b--', label='CDF fit (+=)c')
    ax.plot( x1, cfX_obs, 'm:', lw=3, alpha=0.6, label='CDF data')
    ax.plot( aa, cX_b, 'g-', lw=2, alpha=0.6, label='CDF model (cumsum)')
    ax.legend()
    
    fig, ax = plt.subplots(1,1,figsize=(4,3))
    ax.set_title('CDF model vs data (good)')
    ax.set_xlabel(r'$\theta$ (rads)', fontsize=12)
    ax.set_ylabel('Cumulative distribution', fontsize=12)
    ax.plot( aa, cX_c, 'b--', label='CDF fit (+=)c')
    ax.plot( x1, cfX_obs, 'm:', lw=3, alpha=0.6, label='CDF data')
    ax.legend()
    
    #%% GOF tests: 
    U2, Us, uc, pu2, pus = DFST.watson_GOF( cX_t, alphal=2)
    print('2nd DFst.watson =',U2, Us, uc, pu2, pus)
    
    U2, Us, uc, pu2, pus = DFST.watson_GOF( cX_t-min(cX_t), alphal=2)
    print('3rd DFst.watson =',U2, Us, uc, pu2, pus)
    
    U2, Us, uc, pu2, pus = DFST.watson_GOF( cX_b, alphal=2)
    print('4th DFst.watson =',U2, Us, uc, pu2, pus)
    
    # this is the most correcrt: 
    U2, Us, uc, pu2, pus = DFST.watson_GOF( cX_c, alphal=1)
    print('5th DFst.watson =',U2, Us, uc, pu2, pus)
    
    Vn, pVn = DFST.Kuiper_GOF( cX_b )
    print('1st Kuiper:', Vn, pVn)
    
    Vn, pVn = DFST.Kuiper_GOF( cX_t )
    print('2nd Kuiper:', Vn, pVn)
    
    # this is the most correcrt: 
    Vn, pVn = DFST.Kuiper_GOF( cX_c )
    print('3rd Kuiper:', Vn, pVn)
    
    # for the R2 coefficient: 
    # keep 
    fX_r2 = np.zeros(len(x1),)
    for mu, kap, pii in zip(loc_mle, kap_mle, p_mle):
        temp = stats.vonmises( kap, mu, scal_ )
        fX_r2 += pii*temp.pdf( x1 )
        
    dx = np.diff(x1)
    cX_r2 = np.ones(len(x1),)
    cX_r2[0:-1] = np.cumsum(fX_r2[0:-1]*dx)
    R2 = DFST.myR2( cX_r2, cfX_obs )
    print('R2:', R2)
    
    # keep: 
    DFST.PP_GOF( cX_r2, cfX_obs )
    
    fig, ax = plt.subplots(1,1,figsize=(9,6))
    ax.plot(x1, cX_r2, 'g')
    
    # for the K-S: 
    Dn = DFST.my_KS_GOF_mvM( p_X, dataFrame, alpha=0.10 )
    
    d00 = np.array([1.63, 1.36, 1.22])/np.sqrt(len(x1))
    Dm = abs(cX_r2 - cfX_obs)
    D = max(Dm)
    sqnD = np.sqrt(len(x1))*D
    pval = 1 - DFST.KS_CDF( sqnD )
    print('3rd K-S =',D, sqnD, pval, d00)

    # ---------------------------------------------------------------------- #
    
#    w1, w2 = DFst.ECDF( Int )
#    
#    fig, ax = plt.subplots(1, 1, figsize=(9,3))
#    ax.plot(w1, w2, 'b-', label='data CDF')
    
#    kapp_pr = np.array([kap1, kap2, kap3])
#    locs_pr = np.array([mu1, mu2, mu3])
#    pi_pr = np.array([pp1, pp2, pp3])
#    # plot in the same histogram the approximations:
#    x_ = np.linspace( min(X_samples), max(X_samples), N )
#    X_tot = np.zeros(len(x_),)
#    for mu, kap, pii in zip(locs_pr, kapp_pr, pi_pr):
#        print(mu,kap,pii)
#        X_temp = pii*stats.vonmises.pdf( x_, kap, mu )
#        ax.plot(x_, X_temp, linewidth=2, \
#                label=r'$\mu$ = {}, $\kappa$= {}, p= {} '.format(round(mu,3), round(kap,3), round(pii,3)))
#        X_tot += X_temp
#    ax.plot(x_, X_tot, color='red', linewidth=3, label='fit mixture')
#    ax.set_xlabel(r'$\theta$ (rad)', fontsize=12)
#    ax.set_ylabel('f(x)', fontsize=12)
#    ax.legend(loc=1)

#%%
elif YNtest == 'No':
    print('Real simulation ...')
    
    #im_name = input('Give the name of the image file: ')
    # im_name = "MAX_noepi_C2.png"
    ##im_name = "SR_01.png"
    #csv_name = input('Give the name of the csv file: ')
    # csv_name = "MAX_noepi_C2.csv"
    ##csv_name = "SR_01_FFT.csv"
    
    # if you have a different file path:
    im_path = myPath + 'MAX_20X_Airyscan_6.jpg'
    csv_path = myPath + "MAX_20X_Airyscan_6.csv"
    #
    # im_path = myPath + 'Clipboard.png'
    # csv_path = myPath + "Dir_hist_FFT.csv"
    # im_path = myPath + 'InCos.png'
    # csv_path = myPath + "InCos0pi.csv"
    
    # this is how to split the path from the file name:
    # use it in 
    temp_path, temp_file = os.path.split(im_path)
    print(temp_path)
    print(temp_file)
        
    # number of clusters:
    n_clust = 3
    
    angles, values = dC.imageCVS2Data( csv_path )
    
    c, s, r, rb, al0, al0d = dC.circ_measures(angles, values)
    print(c, s, r, rb, al0, al0d)
    
    values = values + 0.00001
    # ------------- 
    # if you have the angles and values and n_clust, then type the following
    # lines (between the dash-lines) in your code:
    tt1 = time.process_time()
    res_vonMF, mixvM = dC.data2vonMises( angles, values, n_clust, im_path )
    tt2 = time.process_time()
    total_time = tt2 - tt1
    
    DFst.plot_mixs_vonMises_General(mixvM)
    
    DFst.plot_mixs_vonMises_Specific(mixvM, angles)
    
    # check goodness-of-fitness measure R2:
    fhat, Fhat, Fexp = dC.myCDFvonMises( angles, values, res_vonMF );
    R2 = dC.myR2( Fhat, Fexp )
    
    dC.myPPplot( Fhat, Fexp, im_path, n_clust )
    
    print(total_time)
    print(res_vonMF)
    print('R^2 =',R2)
    
    YNplot = 'Yes'
    if YNplot == 'Yes':
        
        n_X = dC.normalizeIntensity( angles, values )
        #n_X_new = n_X.copy()
        #n_X_new[:,1] = n_X_new[:,1] - min(n_X_new[:,1])
        dC.plotMixvMises1X2( res_vonMF, im_path, n_X )
        
    # ------------- 
    print('... finished!')

else:
    print('Test simulation ... ')
    ress = dC.test()
    print('... finished!')

#    # ------------------------------------- # 
#    # if you solve with fsolve: 
#    in_guess = [ p1_, kappa1_, loc1_, p2_, kappa2_, loc2_, kappa3_, loc3_ ]
#    print('METHOD II: - - - - - - without providing the Jacobian - - - - - - ')
#    results = optimize.fsolve( DFfsolve.myF_3vM, in_guess, args=(r_X,Int), \
#                                 full_output=True, xtol=1.49012e-8, maxfev=0 )
#    print('solution with METHOD II = ', results)
#    print('parameters with METHOD II = ', results[0])
#    sol_III = results[0]
#    kappa1_mle = sol_III[1]
#    kappa2_mle = sol_III[4]
#    kappa3_mle = sol_III[6]
#    mu1_mle = sol_III[2]
#    mu2_mle = sol_III[5]
#    mu3_mle = sol_III[7]
#    p1_mle = sol_III[0]
#    p2_mle = sol_III[3]
#    p3_mle = 1. - p1_mle - p2_mle
#    # ------------------------------------- # 
