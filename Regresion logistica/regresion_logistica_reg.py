# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:01:46 2021

@author: Regresion logistica para modelos no lineales 
Ejercicio para decidir el estado de un microship de acuerdoa unos test 
"""

import pandas as pd 
import numpy as np 
from graficar import graficar
import matplotlib.pyplot as plt 
from mapeo_polinomico import mapeo_polinomico
from cg_log_Reg import cg_log_Reg
from limite import limite 
import scipy.optimize as opt


'***______Cargar los datos________***'

datos=pd.read_csv('ex2data2.txt', header=None)
x=datos.values[:,:2]
y=datos.values[:,2:3]
m=len(y)

'***Graficar los datos***'
graficar(datos)
plt.xlabel('Test 1')
plt.ylabel('Test 2')

'***ajustar X creando mas caracteristicas***'

caract_x=mapeo_polinomico(x[:,0].reshape(m,1),x[:,1].reshape(m,1))
(m,n)=caract_x.shape

theta_inicial= np.zeros(n,dtype=int)#debe ser un arreglo para fmin

'parametro de  regularizacion'
landa = 1

'Computar la funcion costo y gradiente; regresion logistica'
[costo,grad]= cg_log_Reg(theta_inicial, caract_x, y, landa)
np.set_printoptions(suppress=True)

print('La funcion costo a theta inicial (ceros) %g'  %costo)
print('Costo esperado (approx): 0.693\n');
print('El gradiente a thetha inicial (ceros) - primeros cinco valores solamente:\n ');
print('\n', grad[:5])
print('Gradientes esperados (approx) -primeros cinco valores solamente:\n' )
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

'***Prueba con otro theta***'

theta_prueba=np.ones(n,dtype=int)
[costo,grad]= cg_log_Reg(theta_prueba, caract_x, y, 10)

print('La funcion costo a theta inicial (unos,lamda=10) %g'  %costo)
print('Costo esperado (approx): 3.16\n');
print('El gradiente a thetha inicial (ceros) - primeros cinco valores solamente:\n ');
print('\n', grad[:5])
print('Gradientes esperados (approx) -primeros cinco valores solamente:\n' )
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.092\n')

'***_______optimizacion_________***'

landa=1

resultado = opt.fmin_tnc(func=cg_log_Reg, x0=theta_inicial, args=(caract_x, y,landa))
theta_min=resultado[0]
'***Grafica del limite de decision***'
datos_exp=pd.DataFrame(np.concatenate([caract_x,y],axis=1))
limite(theta_min,datos_exp)
graficar(datos)
plt.show()





