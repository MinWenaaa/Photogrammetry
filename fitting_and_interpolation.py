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

    X = (np.linalg.inv(M.T@P@M)@M.T@P@Z.T).ravel()

    return X[5]

# 矩阵输出为tif
def output2tif(array, file_path):
    driver = gdal.GetDriverByName('GTiff')
    cols = array.shape[1]
    rows = array.shape[0] 
    dst_ds = driver.Create(file_path, cols, rows, 1, gdal.GDT_Float32)  
    band = dst_ds.GetRasterBand(1)
    band.WriteArray(array)  
    dst_ds.FlushCache()
    dst_ds = None


def linear_fitting(origin_array, origin_dpi, new_dpi):
    origin_height = origin_array.shape[0]
    origin_width = origin_array.shape[1]
    new_height = int(origin_height*origin_dpi/new_dpi)
    new_width = int(origin_width*origin_dpi/new_dpi)
    new_array = np.zeros((new_height, new_width))

    for i in range(new_height):
        for j in range(new_width):
            x = int(j*new_dpi/origin_dpi)
            y = int(i*new_dpi/origin_dpi)
            X = ((j*new_dpi)%origin_dpi)/origin_dpi
            Y = ((i*new_dpi)%origin_dpi)/origin_dpi
            value = origin_array[y:y+2, x:x+2]
            A = np.array([[(1-X)*(1-Y), X*(1-Y)], [(1-X)*Y, X*Y]])
            test = np.sum(A*value)
            new_array[i][j] = test

    return new_array

if __name__=="__main__":
    """
    data_matrix = read_point_data('data/pre-points  - Cloud2.txt')
    result = moving_surface_interpolation(data_matrix, 30, 15)
    np.savetxt('output/moving_curve.csv', result, delimiter=',', fmt='%f')
    # output2tif(result, 'output/moving_curve.tif')
    """

    table = pd.read_csv('output/moving_curve.csv')
    moving = table.to_numpy()

    moving = moving[:,2].reshape((88,55)).T

    linear = linear_fitting(moving, 15, 10)
    df = pd.DataFrame(linear)
    df.to_csv('output/linear.csv', index=False)

    output2tif(linear, 'output\linear.tif')

    