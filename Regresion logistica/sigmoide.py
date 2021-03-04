# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:03:38 2021

@author: Luis Carlos Prada S
Implementacion de la funcion sigmoide g(z) para la regresion logistica 
"""
import numpy as np 
def sigmoide(z):
    
    g = np.zeros(z.shape)
    g=1/(1+np.exp(-z))
    return g 
    

