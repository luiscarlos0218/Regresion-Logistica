# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:16:44 2021

@author: Lusi carlos Prada 
Gradiente y funcion costo con regularizacion 
"""
import numpy as np 
from sigmoide import sigmoide 
def cg_log_Reg(theta,x,y,landa):
    
    m = len(y) 
    n=x.shape[1]
    theta=theta.reshape(len(theta),1)
    grad =np.zeros([theta.size,1])
    
    
    g=sigmoide(np.dot(x,theta))  
    J=(1/m)*(np.sum(np.dot((-y.T),np.log(g))-np.dot((1-y).T,np.log(1-g))))+(landa/(2*m))*np.sum(np.power(theta[1:len(theta)],2))
    
    grad[0]=(1/m)*np.dot((g-y).T,x[:,0])
    
    
    grad[1:n] =(np.dot((x[:,1:n]).T,(g-y)))/m +(landa/m)*(theta[1:n])#%se calcula el gradiente teniendo en cuenta cada una de las columnas de X que representan cada una de las caracteristicas
    
    return J,grad
 

