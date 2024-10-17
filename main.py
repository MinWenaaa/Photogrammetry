import pandas as pd
from spatial_rear_intersection import spatialRearIntersection


m = float(50000)
f = float(153.24)
path = 'input1.xlsx'
df = pd.read_excel(path, engine='openpyxl', dtype=float)
coordination = df.to_numpy()

X = spatialRearIntersection(m, f, coordination)

print(X)