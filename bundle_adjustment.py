import numpy as np
import pandas as pd
from math import cos, sin, sqrt
from rear_frontal_intersection import AandL, spatialRearIntersection, rearAndFrontal

def bundleAdjustment(m, f, GCP_left, GCP_right, GCP_ground, unknown_left, unknown_right):
    # 获取初始值
    EO_left = spatialRearIntersection(m, f, GCP_left, GCP_ground)
    EO_right = spatialRearIntersection(m, f, GCP_right, GCP_ground)
    t = np.hstack((EO_left, EO_right))

    X = rearAndFrontal(m=m, f=f,
        GCP_left=GCP_left, GCP_right=GCP_right, GCP_ground=GCP_ground,
        unknown_left=unknown_left, unknown_right=unknown_right
    )

    count = 0

    while(True):
        count = count + 1
        temp_left = AandL(f, t[0:6], 
            image = np.vstack((GCP_left, unknown_left)), 
            ground = np.vstack((GCP_ground, X))
        )
        temp_right = AandL(f, t[6:12], 
            image = np.vstack((GCP_right, unknown_right)), 
            ground = np.vstack((GCP_ground, X))
        )

        A_1 = temp_left[:,0:6]
        A_2 = temp_right[:,0:6]
        L = np.hstack((temp_left[:,6], temp_right[:,6]))
        A = np.hstack((
            np.vstack((A_1, np.zeros_like(A_1))), 
            np.vstack((np.zeros_like(A_1), A_2)), 
        ))
        
        n = GCP_left.shape[0] + unknown_left.shape[0]
        B = np.zeros((4*n, 3*n))
        for i in range(n):
            B[i, i*3:i*3+3] = -temp_left[i, 0:3]
            B[i+n, i*3:i*3+3] = -temp_left[i+n, 0:3]
            B[i+2*n, i*3:i*3+3] = -temp_right[i, 0:3]
            B[i+3*n, i*3:i*3+3] = -temp_right[i+n, 0:3]
        
        B = B[:, 3*GCP_left.shape[0]:]

        """
        C = np.hstack((A,B))
        dX = (np.linalg.inv(C.T@C)@C.T@L.T).ravel()
                t = t + dX[0:12]
        X = X + dX.reshape(9,3)[8:13,:]
        
        """
        N_11 = A.T @ A
        N_12 = A.T @ B
        N_22 = B.T @ B
        u_1 = A.T @ L
        u_2 = B.T @ L

        dt = np.linalg.inv(N_11 - N_12 @ np.linalg.inv(N_22) @ N_12.T) @ (u_1 - N_12 @ np.linalg.inv(N_22) @ u_2)
        dX = np.linalg.inv(N_22 - N_12.T @ np.linalg.inv(N_11) @ N_12) @ (u_2 - N_12.T @ np.linalg.inv(N_11) @ u_1)
        
        t = t + dt
        X = X + dX.reshape(5,3)
    
        if (dX<0.3e-4).all() and (dt[:3]<0.3e-4).all() and (dt[3:]<1e-5).all():
            break;
    
    print(X)
    print(count)

    return 0

if __name__=="__main__":
    f = 150
    path = 'input1.xlsx'
    df = pd.read_excel(path, engine='openpyxl', dtype=float)
    coordination = df.to_numpy()
    GCP_left = coordination[0:4,0:2] 
    GCP_right = coordination[0:4,2:4]
    GCP_ground = coordination[0:4,4:7]
    unknown_left = coordination[4:9, 0:2]
    unknown_right = coordination[4:9, 2:4]
    bundleAdjustment(10000, 150, GCP_left, GCP_right, GCP_ground, unknown_left, unknown_right)