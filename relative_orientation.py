import numpy as np
import pandas as pd
from spatial_rear_intersection import rotationMatrix

def relativeOrientation(point_left, point_right, f):
    X = np.zeros(5)
    bu = np.mean(point_left[:,0]- point_right[:,0])
    n = point_left.shape[0]

    while(True):
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

        X = X + dX

        if(abs(dX[2])>0.00003 or abs(dX[3])>0.00003 or abs(dX[4])>0.00003):
            continue
        break

    print(X)


df = pd.read_excel('input2.xlsx', engine='openpyxl', dtype=float)
data = df.to_numpy()

point_left = data[:, 0:2]
point_right = data[:, 2:4]

relativeOrientation(point_left, point_right, 150)