#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 10:48:48 2018

@author: df
"""
import numpy as np
import sympy

x1, x2 = sympy.symbols('x_1, x_2')
f_sym = (x1 - 1)**4 + 5*(x2 - 1)**2 - 2*x1*x2
f_lmbda = sympy.lambdify((x1, x2), f_sym, 'numpy')

def func_XY_to_X_Y(f):
    """
    Wrapper for f(X) -> f(X[0], X[1])
    """
    return lambda X: np.array*f(X[0], X[1])

f = func_XY_to_X_Y(f_lmbda)
