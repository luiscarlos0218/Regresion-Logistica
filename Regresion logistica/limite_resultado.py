# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:15:02 2021

@author: Luis Carlos Prada 
Funcion para graficar el limite de decision, resultado de la regresion logistca
"""

#from graficar import graficar 
import matplotlib.pyplot as plt 

def limite_desicion(theta_opt,datos):
    #graficar(datos);
    x= datos.values[:,:2]  #(100,2)
   
    if (x.shape[1] <= 3):
        
        eje_x = [min(x[:,1])-2,  max(x[:,1])+2];
    
        #limite de decision lineal 
        eje_y = (-1/theta_opt[2])*(theta_opt[1]*eje_x + theta_opt[0])
    
        plt.plot(eje_x, eje_y)
        plt.show()
        
       