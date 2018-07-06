#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:02:56 2018

@author: df
"""

# import the packages: 
import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats
import time
import matplotlib.pyplot as plt

# Set up your x values: 
x = np.linspace(0, 100, num=100)

# Set up your observed y values with a known slope (2.4), intersept (5.), 
# and sd (4.): 
yObs = 5. + 2.4*x + np.random.normal(0, 4., 100)

# Define the likelihood funcrtion where params is a list of initial parameter
# estimates: 
def regressLL(params):
    # Resave the initial parameter guesses: 
    b0 = params[0]
    b1 = params[1]
    sd = params[2]
    
    # Calculate the predicted values fromn the parameter guesses: 
    yPred = b0 + b1*x
    
    # Calculate the negative log-likelihood as the negative sum of the log of 
    # a normal PDF where the observed values are normally distributed around
    # the mean (yPred) with a standard deviation of sd: 
    logLik = -np.sum( stats.norm.logpdf( yObs, loc=yPred, scale=sd ) )
    
    # Tell the function to return the NLL (this is what will be minimized):    
    return logLik

# Make a list of initial parameter guesses (b0, b2, sd): 
initParams = [1, 1, 1]

# Run the minimizer: 
results = minimize( regressLL, initParams, method='nelder-mead' )

# Print the results. They should be really close to your actual values: 
print(results.x) 

xx = results.x[0] + results.x[1]*x + stats.norm.rvs(0, results.x[2], 100)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(x, xx, 'r', label='yPred')
ax.plot(x, yObs, 'b', label='yObs')
ax.legend()

