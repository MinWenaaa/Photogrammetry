import numpy as np
import pandas as pd
from math import sqrt
from rear_frontal_intersection import rotationMatrix, frontalIntersection

def relativeOrientation(point_left, point_right, f):
    X = np.zeros(5)
    bu = np.mean(point_left[:,0]- point_right[:,0])
    n = point_left.shape[0]

    count = 0
    while(True):
        count = count+1

        R = rotationMatrix(X[0], X[1], X[2])

        auxiliary_left = np.hstack((point_left, -f*np.ones((n,1))))
        auxiliary_right = (R @ np.hstack((point_right, -f*np.ones((n,1)))).T).T

        bv = bu*X[3]
        bw = bu*X[4]

        N_1 = (bu*auxiliary_right[:,2] - bw*auxiliary_right[:,0])/(auxiliary_left[:,0]*auxiliary_right[:,2]-auxiliary_left[:,2]*auxiliary_right[:,0])
        N_2 = (bu*auxiliary_left[:,2] - bw*auxiliary_left[:,0])/(auxiliary_left[:,0]*auxiliary_right[:,2]-auxiliary_left[:,2]*auxiliary_right[:,0])

        a = - auxiliary_right[:,0] * auxiliary_right[:,1] * N_2 /auxiliary_right[:,2]
        b = -(auxiliary_right[:,2]+auxiliary_right[:,1]*auxiliary_right[:,1]/auxiliary_right[:,2]) * N_2
        c = auxiliary_right[:,0] * N_2
        d = bu * np.ones(n)
        e = - bu * auxiliary_right[:,1] / auxiliary_right[:,2]
        L = N_1 * auxiliary_left[:,1] - N_2 * auxiliary_right[:,1] - bv
        A = np.vstack((a, b, c, d, e)).T

        dX = (np.linalg.inv(A.T@A)@A.T@L.T).ravel()

        print("dX for count", count, '\n', dX)
        X = X + dX

        if(abs(dX[2])>0.00003 or abs(dX[3])>0.00003 or abs(dX[4])>0.00003):
            continue
        break

    V = A@dX - L
    VTPV = sqrt(V@V.T/(2*n-5))
    print("单位权中误差：")
    print(VTPV, '\n', X)

    return np.array([X[0], X[1], X[2], bu*10, bu*X[3]*10, bu*X[4]*10])

def imagePoint2groundPoint(image_left, image_right, RO_elements, f):
    EO_left = np.array([0, 0, 0, 0, 0, 0])
    EO_right = np.hstack((RO_elements[3:6], RO_elements[0:3]))

    groundPoint = np.zeros((image_left.shape[0], 3))

    for i in range(image_left.shape[0]):
        groundPoint[i,:] = frontalIntersection(f, EO_left, EO_right, image_left[i,:], image_right[i,:])
    
    return groundPoint

def absoluteOrientation(auxiliary, geodetic):
    
    X = np.array([0, 0, 0, 1, 0, 0, 0])
    n = auxiliary.shape[0]

    geodetic_copy = geodetic - np.mean(geodetic, axis=0)
    auxiliary_copy = auxiliary - np.mean(auxiliary, axis=0)

    A = np.zeros((n*3, 7))
    for i in range(n): 
        A[i*3,:] = np.array([1,0,0,auxiliary_copy[i,0],-auxiliary_copy[i,2],0,-auxiliary_copy[i,1]]) 
        A[i*3+1,:] = np.array([0,1,0,auxiliary_copy[i,1],0,-auxiliary_copy[i,2], auxiliary_copy[i,0]])
        A[i*3+2,:] = np.array([0,0,1,auxiliary_copy[i,2],auxiliary_copy[i,0],auxiliary_copy[i,1],0])
    
    L = np.zeros(n*3)

    while(True):
        # 常数项与误差方程系数
        R_0 = rotationMatrix(X[4],X[5],X[6])

        for i in range(n):
            L[i*3:i*3+3] = geodetic_copy[i,:] - X[3]*R_0@auxiliary_copy[i,:] - X[0:3]
        
        dX = (np.linalg.inv(A.T@A)@A.T@L.T).ravel()

        Lambda = X[3]
        X = X + dX
        X[3] = Lambda*(1+dX[3])

        if dX[4]<0.00001:
            break
        continue


    return X

def getGeodeticPoint(AO_elements, auxilary):

    R = rotationMatrix(AO_elements[4], AO_elements[5], AO_elements[6])

    n = auxilary.shape[0]
    geodetic_point = np.zeros((n,3))
    for i in range(n):
        geodetic_point[i,:] = AO_elements[3] * R @ auxilary[i,:] + AO_elements[0:3]

    return geodetic_point


if __name__=="__main__":
    
    f = 150
    path = 'input4.xlsx'
    df = pd.read_excel(path, engine='openpyxl', dtype=float)
    coordination = df.to_numpy()
    point_left = coordination[0:4,0:2] 
    point_right = coordination[0:4,2:4]
    point_ground = coordination[0:4,4:7]

    unknown_left = coordination[4:9, 0:2]
    unknown_right = coordination[4:9, 2:4]

    RO_elements = relativeOrientation(np.vstack((point_left, unknown_left)), np.vstack((point_right, unknown_right)), f)

    auxi_ground = imagePoint2groundPoint(point_left, point_right, RO_elements, f)
    unknown_auxi_ground = imagePoint2groundPoint(unknown_left, unknown_right, RO_elements, f)

    AO_elements = absoluteOrientation(auxi_ground, point_ground)
    means_auxi = np.mean(auxi_ground, axis=0)
    means_geodetic = np.mean(point_ground, axis=0)

    result = getGeodeticPoint(AO_elements, unknown_auxi_ground-means_auxi)

    print(result + means_geodetic)
