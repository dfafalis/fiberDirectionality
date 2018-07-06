#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 11:06:54 2018

@author: df
"""

import numpy as np
from scipy import stats
from numpy import i0 # modified Bessel function of the first kind order 0, I_0
from scipy.special import iv # modified Bessel function of first kind, I-v 


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
    scal_ = 0.5
    x_ = np.array(params[0]).T
    i_ = np.array(params[1]).T
#    print(type(x_))
#    print('x_=', x_.shape)
#    print(type(i_))
#    print('i_=', i_.shape)
    
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
    fvm1_ = stats.vonmises.pdf( x_, kappa1_, m1_, scal_ )
    # the 2nd von Mises distribution:
    fvm2_ = stats.vonmises.pdf( x_, kappa2_, m2_, scal_ )
    # the erd von Mises distribution:
    fvm3_ = stats.vonmises.pdf( x_, kappa3_, m3_, scal_ )
        
    # mixture distribution: 
    fm_ = p1_*fvm1_ + p2_*fvm2_ + p3_*fvm3_

    # first derivative wrt weight p1:
    dldp1 = sum( np.multiply( np.divide( np.subtract( fvm1_, fvm3_ ), fm_ ), i_ ) )
        
    # first derivative wrt weight p2:
    dldp2 = sum( np.multiply( np.divide( np.subtract( fvm2_, fvm3_ ), fm_ ), i_ ) )
    
    # first derivative wrt location mu1:=
    dldm1 = (kappa1_*p1_/scal_)*sum( np.multiply( np.multiply( np.divide( fvm1_, fm_ ), \
                                          np.sin((x_ - m1_)/scal_) ), i_ ) )
    
    # first derivative wrt location mu2:=
    dldm2 = (kappa2_*p2_/scal_)*sum( np.multiply( np.multiply( np.divide( fvm2_, fm_ ), \
                                          np.sin((x_ - m2_)/scal_) ), i_ ) )
    
    # first derivative wrt location mu3:=
    dldm3 = (kappa3_*p3_/scal_)*sum( np.multiply( np.multiply( np.divide( fvm3_, fm_ ), \
                                          np.sin((x_ - m3_)/scal_) ), i_ ) )
    
    # first derivative wrt concentration kappa1:
    Ak1 = ( iv(1.0, kappa1_) / i0(kappa1_) )    
    dldk1 = p1_*sum( np.multiply( np.multiply( np.divide( fvm1_, fm_ ), \
                                 ( np.cos((x_ - m1_)/scal_) - Ak1 ) ), i_ ) )
    
    # first derivative wrt concentration kappa2:
    Ak2 = ( iv(1.0, kappa2_) / i0(kappa2_) )    
    dldk2 = p2_*sum( np.multiply( np.multiply( np.divide( fvm2_, fm_ ), \
                                 ( np.cos((x_ - m2_)/scal_) - Ak2 ) ), i_ ) )
    
    # first derivative wrt concentration kappa3:
    Ak3 = ( iv(1.0, kappa3_) / i0(kappa3_) )    
    dldk3 = p3_*sum( np.multiply( np.multiply( np.divide( fvm3_, fm_ ), \
                                 ( np.cos((x_ - m3_)/scal_) - Ak3 ) ), i_ ) )

    F = [ dldp1, dldk1, dldm1, \
          dldp2, dldk2, dldm2, \
                 dldk3, dldm3 ]
    
    return F


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
    scal_ = 0.5
    x_ = np.array(params[0]).T
    i_ = np.array(params[1]).T
    
    # the unknown parameters:
    p1_ = theta[0]
    kappa1_ = theta[1]
    m1_ = theta[2]
    kappa2_ = theta[3]
    m2_ = theta[4]
    p2_ = 1.0 - p1_ 
    
    # the 1st von Mises distribution on semi-circle:
    fvm1_ = stats.vonmises.pdf( x_, kappa1_, m1_, scal_ )
    # the 2nd von Mises distribution on semi-circle:
    fvm2_ = stats.vonmises.pdf( x_, kappa2_, m2_, scal_ )
        
    # mixture distribution: 
    fm_ = p1_*fvm1_ + p2_*fvm2_ 

    # first derivative wrt weight p1:
    dldp1 = sum( np.multiply( np.divide( np.subtract( fvm1_, fvm2_ ), fm_ ), \
                              i_ ) )
        
    # first derivative wrt location mu1:=
    dldm1 = (kappa1_*p1_/scal_)*sum( np.multiply( np.multiply( np.divide( fvm1_, fm_ ), \
                                         np.sin((x_ - m1_)/scal_) ), i_ ) )
    
    # first derivative wrt location mu2:=
    dldm2 = (kappa2_*p2_/scal_)*sum( np.multiply( np.multiply( np.divide( fvm2_, fm_ ), \
                                         np.sin((x_ - m2_)/scal_) ), i_ ) )
    
    # first derivative wrt concentration kappa1:
    Ak1 = ( iv(1.0, kappa1_) / i0(kappa1_) )    
    dldk1 = p1_*sum( np.multiply( np.multiply( np.divide( fvm1_, fm_ ), \
                                 ( np.cos((x_ - m1_)/scal_) - Ak1 ) ), i_ ) )
    
    # first derivative wrt concentration kappa1:
    Ak2 = ( iv(1.0, kappa2_) / i0(kappa2_) )    
    dldk2 = p2_*sum( np.multiply( np.multiply( np.divide( fvm2_, fm_ ), \
                                 ( np.cos((x_ - m2_)/scal_) - Ak2 ) ), i_ ) )
    
    F = [ dldp1[0], dldk1[0], dldm1[0], dldk2[0], dldm2[0] ]
    
    return F

