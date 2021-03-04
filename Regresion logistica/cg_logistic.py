# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:55:02 2021

@author: Luis Carlos Prada 
Funcion para calcular el costo y el gradiente para la regresion logistica, 
se usa como funcion la funcion sigmoide 
""" 
import numpy as np 
from sigmoide import sigmoide 

def cg_logistic(theta,x,y):
   
    'valores iniciales'
    m =len(y) #numero de ejemplos de entrenamiento
    theta=theta.reshape(len(theta),1)
   # grad = np.zeros(theta.shape)
    
    
    'funcion sigmoide'
     
    
    
    g=sigmoide(np.dot(x,theta))
    J=(1/m)*(np.sum(np.dot((-y.T),np.log(g))-np.dot((1-y).T,np.log(1-g))))
    #x=pd.DataFrame(x)
   
    grad =np.dot(x.T,(g-y))/m
   # for gr in range(theta.size):
        
       
    #    gradiente[gr]=(1/m)*sum((g-y)*np.array(x[gr]).reshape(m,1))
    
    return J,grad
