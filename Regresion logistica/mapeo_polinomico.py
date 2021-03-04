# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:51:56 2021

@author: Luis Carlos Prada 
Mapeo de caracteristicas para mejores modelos  y evitar la alta bias o underfitting
"""

import pandas as pd 
import numpy as np 
def mapeo_polinomico(x1,x2):
    datos=np.concatenate([x1,x2],axis=1)
    datos=pd.DataFrame(datos)
    
    '''Returns a new feature array with more features, comprising of 
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Inputs X1, X2 must be the same size'''
    
    grado_pol = 7 #que enrealidad seria hasta 6 grado por comenzar en 0 la indezacion
    features=pd.DataFrame()
    for i in  range(grado_pol):
        for j in range(i+1):
            features[features.shape[1]+1] = datos[0]**(i-j)*datos[1]**j
            
    return features.values
        

