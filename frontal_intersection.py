import numpy as np
import pandas as pd
from spatial_rear_intersection import spatialRearIntersection, rotationMatrix


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



df = pd.read_excel('input2.xlsx', engine='openpyxl', dtype=float)
data = df.to_numpy()
m = 10000
f = 150

GCP_left = data[0:4, 0:2]
GCP_right = data[0:4, 2:4]
GCP_ground = data[0:4, 4:7]
unknownPoint_left = data[4:9, 0:2]
unknownPoint_right = data[4:9, 2:4]

exteriorOrientation_left = spatialRearIntersection(m, f, GCP_left, GCP_ground)
exteriorOrientation_right = spatialRearIntersection(m, f, GCP_right, GCP_ground)


#for i in range(4):
#    print(frontalIntersection(f, exteriorOrientation_left, exteriorOrientation_right, GCP_left[i, :], GCP_right[i,:]))

for i in range(5):
    print(frontalIntersection(f, exteriorOrientation_left, exteriorOrientation_right, unknownPoint_left[i, :], unknownPoint_right[i,:]))