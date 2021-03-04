# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:37:42 2021

@author: Luis Carlos Prada Socha 
Regresion Logistica o regresion de clasificacion para 2 variables 

Ejemplo: Se requiere predecir la probabilidad de que un estudiante pase a la U 
conociendo los datos de dos de sus examenes, se tiene un set de entrenamiento con casos de exito y no exito

"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from graficar import graficar
from cg_logistic import cg_logistic
from limite import limite
from sigmoide import sigmoide 
from prediccion import prediccion
import scipy.optimize as opt

'***__________Cargar los datos_________***'

datos=pd.read_csv('ex2data1.txt', header=None)
x= datos.values[:,:2]  #(100,2)
y = datos.values[:,2:3] #(100,1)

'***__________Visualizacion de datos_________****'

graficar(datos)


'***_____Gradiente descendiente_____***'
'Esta es la opcion 1 y no tan recomendada de buscar los parametros de entrenamiento'
(m,n)=x.shape
unos=np.ones([m,1])
x=np.concatenate((unos,x),axis=1)

theta_inicial = np.zeros([n+1],dtype=int) #debo darle un arreglo(n,) para que funcione fmin
[costo,gradiente] = cg_logistic(theta_inicial,x, y)

print(f'La funcion costo para thetas iniciales ceros es {costo}')
print('El valor esperado aproximado es 0.693\n')
print('El gradiente para los thetas iniciales  0 es : \n')
print(gradiente)
print('Los valores esperados aproximados son :\n -0.1000\n -12.0092\n -11.2628\n')

'Probamos con otros theta'
theta_prueba = np.array([[-24],[0.2],[0.2]]) # debe ser un vector columna para que le sirva a la funcion 
[costo,gradiente] = cg_logistic(theta_prueba,x, y)

print(f'La funcion costo para thetas iniciales prueba es {costo}');
print('El valor esperado aproximado es 0.218\n');
print('El gradiente para los thetas iniciales  prueba es : \n');
print(gradiente)
print('Los valores esperados aproximados son :\n 0.043\n 2.566\n 2.647\n')

'***_______Implementacion de metodo de optimizacion__________***'

resultado = opt.fmin_tnc(func=cg_logistic, x0=theta_inicial, args=(x, y))
theta_min=resultado[0]

print(f'La funcion costo encontrada por fmin es: {costo}')
print('El valor esperado aproximado es: 0.203\n');
print('theta por fmin: \n');
print(theta_min)
print('Thetas esperados aproximados:\n')
print(' -25.161\n 0.206\n 0.201\n')

'Grafica del limite de desicion '

limite(theta_min,datos)
graficar(datos)
plt.show()

'Prediccion'

probabilidad = sigmoide(np.array(np.dot([1,45,85],theta_min.T)));
print(f'Para un estudiante con puntajes de 45 y 85 se predice una probabilidad de admision de {probabilidad}')
print('Valor esperado: 0.775 +/- 0.002\n\n')

'Precision del algoritmo'
p = prediccion(theta_min, x)
y=datos[2]
precision=pd.concat([y,p],axis=1)
precision.columns=['real','prediccion']
valor_precision=np.sum(precision['real']==precision['prediccion'])

print(f'La precision del algoritmo es de:{valor_precision}')
print('La precision esperada es de (aprox): 89.0\n');
print('\n');


