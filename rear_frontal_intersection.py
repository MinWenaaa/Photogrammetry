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

# 求系数矩阵
def AandL(f, EO, image, ground):
    n = image.shape[0]
    R = rotationMatrix(EO[3], EO[4], EO[5])
    bar = (ground-np.tile(EO[0:3], (n, 1))) @ R
    L = np.hstack((image[:,0] + f * bar[:,0] / bar[:,2], image[:, 1] + f * bar[:,1] / bar[:,2]))

    a = np.hstack(( (np.tile(f*R[:,0],(n,1)).T + np.outer(R[:,2], image[:,0]))/np.tile(bar[:,2], (3,1)),
        (np.tile(f*R[:,1],(n,1)).T + np.outer(R[:,2], image[:,1]))/np.tile(bar[:,2], (3,1)) )).T
    b = np.hstack((image[:, 1]*sin(EO[4]) - cos(EO[4])*(f*cos(EO[5])+(image[:, 0]*cos(EO[5])-image[:, 1]*sin(EO[5]))*image[:, 0]/f),
        - image[:, 0]*sin(EO[4]) - cos(EO[4]) * (image[:,1]*(image[:,0]*cos(EO[5])-image[:,1]*sin(EO[5]))/f - f*sin(EO[5]))))
    c = np.hstack((- f*sin(EO[5]) - ((image[:,0]*sin(EO[5])+image[:, 1]*cos(EO[5]))*image[:, 0]/f),
        - f*cos(EO[5]) - ((image[:,0]*sin(EO[5])+image[:, 1]*cos(EO[5]))*image[:, 1]/f)))
    d = np.hstack((image[:,1], -image[:,0]))

    return np.hstack((a, np.array([b, c, d, L]).T,))

def spatialRearIntersection(m, f, image, ground):

    X = np.array([np.mean(ground[:,0]), np.mean(ground[:,1]), m*f, 0, 0, 0])

    while(True):

        temp = AandL(f, X, image, ground)
        A = temp[:,0:6]
        L = temp[:,6]

        dX = (np.linalg.inv(A.T@A)@A.T@L.T).ravel()

        X = X+dX

        if(abs(dX[0])>0.001 or abs(dX[1])>0.001 or abs(dX[2])>0.001 or abs(dX[3])>0.00003 or abs(dX[4])>0.00003 or abs(dX[5])>0.00003):
            continue
        break

    print(A@dX)
    print(L)

    V = A@dX - L
    VTPV = sqrt(V@V.T/2)
    #print('单位权中误差：\n', VTPV)

    return X

def frontalIntersection(f, exteriorOrientation_left, exteriorOrientation_right, unknownPoint_left, unknownPoint_right):

    R_left = rotationMatrix(exteriorOrientation_left[3], exteriorOrientation_left[4], exteriorOrientation_left[5])
    R_right = rotationMatrix(exteriorOrientation_right[3], exteriorOrientation_right[4], exteriorOrientation_right[5])

    auxiliary_left = R_left @ np.array([unknownPoint_left[0], unknownPoint_left[1], -f])
    auxiliary_right = R_right @ np.array([unknownPoint_right[0], unknownPoint_right[1], -f])

    B = exteriorOrientation_right[0:3] - exteriorOrientation_left[0:3]

    N_1 = (B[0]*auxiliary_right[2] - B[2]*auxiliary_right[0]) / (auxiliary_left[0]*auxiliary_right[2] - auxiliary_right[0]*auxiliary_left[2])
    N_2 = (B[0]*auxiliary_left[2] - B[2]*auxiliary_left[0]) / (auxiliary_left[0]*auxiliary_right[2] - auxiliary_right[0]*auxiliary_left[2])

    ground_1 = exteriorOrientation_left[0:3] + N_1 * auxiliary_left
    ground_2 = exteriorOrientation_right[0:3] + N_2 * auxiliary_right

    ground_1[1] = (ground_1[1]+ground_2[1])/2

    return ground_1

def rearAndFrontal(m, f, GCP_left, GCP_right, GCP_ground, unknown_left, unknown_right):
    EO_left = spatialRearIntersection(m, f, GCP_left, GCP_ground)
    EO_right = spatialRearIntersection(m, f, GCP_right, GCP_ground)

    #print(EO_left)
    #print(EO_right)

    n = unknown_left.shape[0]

    result_Ground = np.zeros((n,3))
    for i in range(n):
        result_Ground[i,:] = frontalIntersection(f=f, 
            exteriorOrientation_left=EO_left, 
            exteriorOrientation_right=EO_right, 
            unknownPoint_left = unknown_left[i, :], 
            unknownPoint_right = unknown_right[i,:]
        )

    return result_Ground

if __name__=="__main__":
    df = pd.read_excel('input2.xlsx', engine='openpyxl', dtype=float)
    data = df.to_numpy()
    m = 10000
    f = 150

    GCP_left = data[0:4, 0:2]
    GCP_right = data[0:4, 2:4]
    GCP_ground = data[0:4, 4:7]
    unknownPoint_left = data[4:9, 0:2]
    unknownPoint_right = data[4:9, 2:4]

    rearAndFrontal(m, f, GCP_left, GCP_right, GCP_ground, unknownPoint_left, unknownPoint_right)