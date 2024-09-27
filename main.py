import numpy as np
import pandas as pd
from math import cos, sin

m = float(input("m："))
f = float(input("f："))
path = input("坐标文件路径（一般为'input1.xlsx')：")

df = pd.read_excel(path, engine='openpyxl', dtype=float)
coordination = df.to_numpy()

phi = 0
omega = 0
kappa = 0
Z_S0 = m*f
X_S0 = np.mean(coordination[:, 2])
Y_S0 = np.mean(coordination[:, 3])

X = np.array([X_S0, Y_S0, Z_S0, phi, omega, kappa])

count =0

while(True):

    R = np.array([
        [cos(X[3])*cos(X[5])-sin(X[3])*sin(X[4])*sin(X[5]), -cos(X[3])*sin(X[5]) - sin(X[3])*sin(X[4])*cos(X[5]), -sin(X[3])*cos(X[4])], 
        [cos(X[4])*sin(X[5]), cos(X[4])*cos(X[5]), -sin(X[4])], 
        [sin(X[3])*cos(X[5]) + cos(X[3])*sin(X[4])*sin(X[5]), -sin(X[3])*sin(X[5]) + cos(X[3])*sin(X[4])*cos(X[5]), cos(X[3])*cos(X[4])]
    ])

    #print('R:\n')
    #print(R)
    
    bar = np.zeros((4,3))
    for i in range(4):
        bar[i, 0] = R[:,0]@(coordination[i, 2:5]-X[0:3]).T
        bar[i, 1] = R[:,1]@(coordination[i, 2:5]-X[0:3]).T
        bar[i, 2] = R[:,2]@(coordination[i, 2:5]-X[0:3]).T

    L = np.zeros((1, 8))
    for i in range(4):
        L[0, i*2] = coordination[i, 0] + f * bar[i,0] / bar[i,2]
        L[0, i*2+1] = coordination[i, 1] + f * bar[i,1] / bar[i,2]
    #print('L:\n')
    #print(L)

    A = np.zeros((8,6), dtype=float)
    for i in range(4):
        A[i*2, 0:3] = (f*R[:,0] + R[:,2]*coordination[i, 0]) / bar[i,2]
        A[i*2+1, 0:3] = (f*R[:,1] + R[:,2]*coordination[i, 1]) / bar[i,2]

        A[i*2, 3] = coordination[i, 1]*sin(X[4]) - cos(X[4])*(f*cos(X[5])+(coordination[i, 0]*cos(X[5])-coordination[i, 1]*sin(X[5]))*coordination[i, 0]/f)
        A[i*2, 4] = - f*sin(X[5]) - ((coordination[i, 0]*sin(X[5])+coordination[i, 1]*cos(X[5]))*coordination[i, 0]/f)
        A[i*2, 5] = coordination[i, 1]
        A[i*2+1, 3] = - coordination[i, 0]*sin(X[4]) - cos(X[4]) * (coordination[i,1]*(coordination[i,0]*cos(X[5])-coordination[i,1]*sin(X[5]))/f - f*sin(X[5]))
        A[i*2+1, 4] = - f*cos(X[5]) - ((coordination[i, 0]*sin(X[5])+coordination[i, 1]*cos(X[5]))*coordination[i, 1]/f)
        A[i*2+1, 5] = - coordination[i, 0]

    #print('B:\n')
    #print(A)

    dX = (np.linalg.inv(A.T@A)@A.T@L.T).ravel()

    print('dX:\n')
    print(dX)
    X = X+dX
    count=count+1
    if(abs(dX[0])>0.001):
        continue
    print('count:',count)
    break


print(X)