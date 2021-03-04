# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:58:37 2021

@author: Luis Carlos Prada 
Codigo para vuisualizar los datos de la regresion logistica 
"""

import matplotlib.pyplot as plt 


def graficar(datos):
    
    
    positivos = datos[datos[2]==1]; 
    negativos = datos[datos[2]==0];


    plt.plot(positivos[0],positivos[1],'+',color='r',label='Admitidos')
    plt.plot(negativos[0],negativos[1],'.',color='y',label='No Admitidos')
    plt.xlabel('Examen 1')
    plt.ylabel('Examen 2')
    plt.legend()
    
    

