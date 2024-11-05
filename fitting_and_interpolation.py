from osgeo import gdal
import pandas as pd
import numpy as np

def read_point_data(filepath):
    with open(filepath, 'r') as file:
        data_str = file.read()
        data_list = data_str.strip().split('\n')
        data_points = [list(map(float, point.split())) for point in data_list]

        data_matrix = np.array(data_points)
    return data_matrix

def moving_surface_interpolation(points, r, d):

    minX = min(points[:,0])
    minY = min(points[:,1])
    X = max(points[:,0]) - minX
    Y = max(points[:,1]) - minY
    nX = int(X/d) + 2
    nY = int(Y/d) + 2

    result = np.zeros((nX*nY, 3))

    for i in range(nX):
        for j in range(nY):
            # 以r为半径搜索相邻点
            point = np.array([minX+i*d, minY+j*d])
            result[i*nY+j,:2] = point
            nearby_point = np.array([0, 0, 0])

            temp_r = r
            while True:
                
                nearby_point = np.array([0, 0, 0])

                for uncheck in points:
                    if (abs(point-uncheck[:2])<temp_r).all and sum((point-uncheck[:2])**2)<temp_r*temp_r:
                        nearby_point = np.vstack((nearby_point, np.array([uncheck])))
                
                if nearby_point.shape[0]>10:
                    break
                temp_r = temp_r+5
             
            result[i*nY+j,2] = moving_surface(nearby_point[1:], point)
        print('line', i)

    return result

def moving_surface(nearby_points, target_point):

    n = nearby_points.shape[0]
    P = np.zeros((n,n))

    nearby_points[:,:2] = nearby_points[:,:2] - target_point
    for i in range(n):
        P[i,i] = 1/sum((target_point-nearby_points[i,:2])**2)
    
    M = np.vstack((nearby_points[:,0]**2, nearby_points[:,0]*nearby_points[:,1], nearby_points[:,1]**2, nearby_points[:,0], nearby_points[:,1], np.ones(n))).T
    Z = nearby_points[:,2]

    dX = (np.linalg.inv(M.T@P@M)@M.T@P@Z.T).ravel()

    return dX[5]

def output2tif(array, file_path):
    driver = gdal.GetDriverByName('GTiff')
    cols = array.shape[1]
    rows = array.shape[0] 
    dst_ds = driver.Create(file_path, cols, rows, 1, gdal.GDT_Float32)  
    band = dst_ds.GetRasterBand(1)
    band.WriteArray(array)  
    dst_ds.FlushCache()
    dst_ds = None

if __name__=="__main__":
    """
    data_matrix = read_point_data('data/pre-points  - Cloud2.txt')
    result = moving_surface_interpolation(data_matrix, 30, 15)
    np.savetxt('output/moving_curve.csv', result, delimiter=',', fmt='%f')
    # output2tif(result, 'output/moving_curve.tif')
    """

    table = pd.read_csv('output/moving_curve.csv')
    result = table.to_numpy()

    result = result[:,2].reshape((88,55)).T

    output2tif(result, 'output\moving_curve.tif')

    