from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

class single_channel_image:
    """
    单通道图片类
    """
    def __init__(self, file_path: str):
        dataset = gdal.Open(file_path)
        self.data = dataset.GetRasterBand(1).ReadAsArray().astype(np.int32)
        self.shape = self.data.shape

    def getpoint_features(self, threshold: int):
        """
        获取单张像片的特征点表
        """
        self.point_features = []
        for i in range(2, self.shape[0]-2):
            for j in range(2, self.shape[1]-2):
                value = self.moravec(i, j)
                if value > threshold:
                    self.point_features.append((i, j, value)) # 前两位是坐标，第三位是算子的值，第四位用于确定该点是否保留
        self.filter(100, 50)
        return
    
    def filter(self, window: int, gap: int):
        """
        一个筛选算法，滑动图窗的同时给每个特征点计分；
        如果特征点moravec值是图窗中最大的，则加分，否则减分；
        记录所有点的分值，分值大于0的特征点视为有效；
        这个算法可以在保证特征点的moravec值较大的同时兼顾点的均匀分布
        """
        m = list(range(0, self.shape[0]-window, gap))
        n = list(range(0, self.shape[1]-window, gap))
        filter_num = np.zeros(len(self.point_features)).astype(np.float16)
        for i in m:
            for j in n:
                # window_max = 0
                # max_index = 0
                indexies = []   # 图窗中所有特征点的索引
                feature_list = []
                for k in range(len(self.point_features)):
                    r = self.point_features[k][0]
                    c = self.point_features[k][1]
                    if r<i or r>i+window or c<j or c>j+window:
                        continue
                    indexies.append(k)
                    feature_list.append(self.point_features[k][2])
                if not indexies:
                    continue
                max_index = indexies[feature_list.index(max(feature_list))]
                for index in indexies:
                    if index != max_index:
                        filter_num[index] -= 2.5
                    filter_num[index] += 2
        filter_index = [i for i, value in enumerate(filter_num) if value > 0]
        feature = [self.point_features[i] for i in filter_index]
        self.point_features = feature

    def moravec(self, x, y):
        deta = self.data[x-2:x+3, y-2:y+3]
        return moravec(deta)

def moravec(data):
    """
    输入图窗数据，返回moravec值
    """
    window = data.shape[0]
    c = int((window-1)/2)
    V = [0, 0, 0, 0]
    for i in range(window-1):
        V[0] += (data[i, c]-data[i+1, c])**2
        V[1] += (data[i,i]-data[i+1, i+1])**2
        V[2] += (data[c,i]-data[c, i+1])**2
        V[3] += (data[window-1-i, i]-data[window-2-i, i+1])**2   
    return min(V)

def CC_matching1(left :single_channel_image, right :single_channel_image):
    """
    书上的算法，计算左右相片特征点的相关系数
    """
    matched_points = []
    for p_left in left.point_features:
        for p_right in right.point_features:
            if abs(p_left[0]-p_right[0]) + abs(p_left[1]-p_right[1]) > 100:
                continue
            data1 = left.data[p_left[0]-2:p_left[0]+3, p_left[1]-2:p_left[1]+3]
            data2 = right.data[p_right[0]-2:p_right[0]+3, p_right[1]-2:p_right[1]+3]
            cc = correlation_coefficient(data1, data2)
            # if cc>max_coefficient:
            #     max_coefficient = cc
            #     p_match = p_right
            
            if cc>0.9:
                # print(p_left - p_right)
                matched_points.append(p_left + p_right)
    return matched_points

def correlation_coefficient(data1, data2):
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    rho = np.sum((data1-mean1)*(data2-mean2)) / np.sqrt(np.sum((data1-mean1)**2) * np.sum((data2-mean2)**2))
    return rho

def draw(left:single_channel_image, right:single_channel_image, matching_point):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    x1, y1, _ = zip(*left.point_features)
    axs[0].imshow(left.data, cmap='gray')
    axs[0].scatter(y1, x1, s=2, color='red')
    axs[0].set_title('left')
    axs[0].axis('off')

    x2, y2, _ = zip(*right.point_features)
    axs[1].imshow(right.data, cmap='gray')
    axs[1].scatter(y2, x2, s=2, color='red')
    axs[1].set_title('right')
    axs[1].axis('off')

    x2, y2, _, x1, y1, _ = zip(*matching_point)
    for i in range(len(x1)):
        line = ConnectionPatch(xyA=(y2[i],x2[i]), xyB=(y1[i],x1[i]), coordsA="data", coordsB="data",axesA=axs[1], axesB=axs[0], color="b")
        fig.add_artist(line)

    plt.show()

if __name__=="__main__":
    threshold = 14000
    window = 5
    left = single_channel_image("data/left.tif")
    right = single_channel_image("data/right.tif")
    right.getpoint_features(threshold)
    left.getpoint_features(threshold)
    matching_point = CC_matching1(right, left)
    draw(left, right, matching_point)