import numpy as np
import pandas as pd
from math import cos, sin, sqrt

# 求旋转矩阵
def rotationMatrix(phi, omega, kappa):
    R = np.array([
            [cos(phi)*cos(kappa)-sin(phi)*sin(omega)*sin(kappa), -cos(phi)*sin(kappa) - sin(phi)*sin(omega)*cos(kappa), -sin(phi)*cos(omega)], 
            [cos(omega)*sin(kappa), cos(omega)*cos(kappa), -sin(omega)], 
            [sin(phi)*cos(kappa) + cos(phi)*sin(omega)*sin(kappa), -sin(phi)*sin(kappa) + cos(phi)*sin(omega)*cos(kappa), cos(phi)*cos(omega)]
        ])
    
    return R



def spatialRearIntersection(m, f, image, ground):

    Z_S0 = m*f
    X_S0 = np.mean(ground[:,0])
    Y_S0 = np.mean(ground[:,1])

    X = np.array([X_S0, Y_S0, Z_S0, 0, 0, 0])

    count = 0

    while(True):

        #print(count, '\n')

        R = rotationMatrix(X[3], X[4], X[5])

        bar = np.zeros((4,3))
        for i in range(4):
            bar[i, 0] = R[:,0]@(ground[i, :]-X[0:3]).T
            bar[i, 1] = R[:,1]@(ground[i, :]-X[0:3]).T
            bar[i, 2] = R[:,2]@(ground[i, :]-X[0:3]).T

        L = np.zeros((1, 8))
        for i in range(4):
            L[0, i*2] = image[i, 0] + f * bar[i,0] / bar[i,2]
            L[0, i*2+1] = image[i, 1] + f * bar[i,1] / bar[i,2]

        A = np.zeros((8,6), dtype=float)
        for i in range(4):
            A[i*2, 0:3] = (f*R[:,0] + R[:,2]*image[i, 0]) / bar[i,2]
            A[i*2+1, 0:3] = (f*R[:,1] + R[:,2]*image[i, 1]) / bar[i,2]

            A[i*2, 3] = image[i, 1]*sin(X[4]) - cos(X[4])*(f*cos(X[5])+(image[i, 0]*cos(X[5])-image[i, 1]*sin(X[5]))*image[i, 0]/f)
            A[i*2, 4] = - f*sin(X[5]) - ((image[i, 0]*sin(X[5])+image[i, 1]*cos(X[5]))*image[i, 0]/f)
            A[i*2, 5] = image[i, 1]
            A[i*2+1, 3] = - image[i, 0]*sin(X[4]) - cos(X[4]) * (image[i,1]*(image[i,0]*cos(X[5])-image[i,1]*sin(X[5]))/f - f*sin(X[5]))
            A[i*2+1, 4] = - f*cos(X[5]) - ((image[i, 0]*sin(X[5])+image[i, 1]*cos(X[5]))*image[i, 1]/f)
            A[i*2+1, 5] = - image[i, 0]


        dX = (np.linalg.inv(A.T@A)@A.T@L.T).ravel()

        count = count + 1

        #print('dX for count ',count)
        #print(dX)
        X = X+dX
        #print(X)
        if(abs(dX[0])>0.001 or abs(dX[1])>0.001 or abs(dX[2])>0.001 or abs(dX[3])>0.00003 or abs(dX[4])>0.00003 or abs(dX[5])>0.00003):
            continue
        break

    V = A@dX - L
    VTPV = sqrt(V@V.T/2)

    return np.hstack((X, VTPV))

