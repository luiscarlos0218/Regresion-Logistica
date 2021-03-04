# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 22:06:52 2021

@author: Luis carlos Prada 
Predecir precision del algoritmo regresion logistica, si g es menor que 0.5 o 
en otras palabras h menor a 0 da como prediccion 0 sino 1  
"""
import numpy as np
import pandas as pd 
def prediccion(theta_opt,x):
    
    m=len(x)
    p=np.zeros(m)
    h=np.dot(x,theta_opt.T)
    p=pd.Series(h)
    p.loc[p<0] = 0
    p.loc[p>0]=1
    return p 