# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 19:19:25 2021

@author: Luis Carlos Prada 
Funcion para graficar el limite de decision, resultado de la regresion logistca
"""

from graficar import graficar 
import matplotlib.pyplot as plt 
import numpy as np 
from mapeo_polinomico import mapeo_polinomico


def limite(theta_opt,datos):
    
    plt.figure()
    x= datos.values[:,:-1]  
    
    if (x.shape[1] <= 3):
        
        eje_x = np.array([min(x[:,1])-2,  max(x[:,1])+2])
    
        #limite de decision lineal 
        eje_y = (-1/theta_opt[2])*((theta_opt[1]*eje_x + theta_opt[0]))
    
        plt.plot(eje_x, eje_y,label='limite_Decision')
        
    else:
        #Definimos el rango de la grilla 
        nd=50 #numero de datos
        eje_1 =np.linspace(-1, 1.5,nd )
        eje_2= np.linspace(-1, 1.5, nd)
        eje_z =np.zeros([len(eje_1), len(eje_2)])
        
        for i in range(len(eje_1)):
                       
            for j in range(len(eje_2)):
                x1=np.array(eje_1[i]).reshape(1,1)
                x2=np.array(eje_2[j]).reshape(1,1)
                mapeo = mapeo_polinomico(x1, x2)
                eje_z[i,j]=np.dot(mapeo,theta_opt.reshape(len(theta_opt),1))
        
    
        #eje_z = np.transpose(eje_z) # important to transpose z before calling contour
    
        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        plt.contour(eje_1, eje_2, eje_z,0,label='limite decision')
   

        