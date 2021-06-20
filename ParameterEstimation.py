# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


# 定义高斯函数（二维正态分布），计算概率p(x|w)。（[xi,yi],某类的中心[xm,ym]，协方差）
def Gaussian_function(x, mean, cov):
    det_cov = np.linalg.det(cov)  # 计算方差矩阵的行列式
    inv_cov = np.linalg.inv(cov)  # 计算方差矩阵的逆
    # 计算概率p(x|w)
    p = 1 / (2 * np.pi * np.sqrt(det_cov)) * np.exp(-0.5 * np.dot(np.dot((x - mean), inv_cov), (x - mean)))
    return p


# 生成正态分布数据
def Generate_Sample_Gaussian(mean, cov, P, label):
    '''
        mean 为均值向量
        cov 为方差矩阵a
        P 为单个类的先验概率
        return 单个类的数据集
    '''
    temp_num = round(1000 * P)  # 先验概率*1000，四舍五入
    x, y = np.random.multivariate_normal(mean, cov, temp_num).T
    z = np.ones(temp_num) * label
    X = np.array([x, y, z])
    return X.T


# 根据不同先验生成不同的数据集
def Generate_DataSet(mean, cov, P):
    # 按照先验概率生成正态分布数据
    # 返回所有类的数据集
    X = []
    label = 1
    for i in range(3):
        # 把此时类i对应的数据集加到已有的数据集中
        X.extend(Generate_Sample_Gaussian(mean[i], cov, P[i], label))
        label += 1
        i = i + 1
    return X


# 画出不同先验对应的散点图
def Generate_DataSet_plot(mean, cov, P):
    xx = []
    label = 1
    for i in range(3):
        xx.append(Generate_Sample_Gaussian(mean[i], cov, P[i], label))
        label += 1
        i = i + 1
    # 画图
    plt.figure()
    for i in range(3):
        plt.plot(xx[i][:, 0], xx[i][:, 1], '.', markersize=4.)
        plt.plot(mean[i][0], mean[i][1], 'r*')
    plt.show()
    return xx


# 似然率测试规则（不需要先验概率）
def Likelihood_Test_Rule(X, mean, cov):
    class_num = mean.shape[0]  # 类的个数
    num = np.array(X).shape[0]
    error_rate = 0
    for i in range(num):
        p_temp = np.zeros(3)
        for j in range(class_num):
            p_temp[j] = Gaussian_function(X[i][0:2], mean[j], cov)  # 计算样本i决策到j类的概率
        p_class = np.argmax(p_temp) + 1  # 得到样本i决策到的类
        if p_class != X[i][2]:
            error_rate += 1
    return error_rate / num


# 最大后验概率规则（样本，每类中心，协方差矩阵，先验概率）
def Max_Posterior_Rule(X, mean, cov, P):
    class_num = mean.shape[0]  # 类的个数
    num = np.array(X).shape[0]
    error_rate = 0
    for i in range(num):
        p_temp = np.zeros(3)
        for j in range(class_num):
            p_temp[j] = Gaussian_function(X[i][0:2], mean[j], cov) * P[j]  # 计算样本i是j类的后验概率
        p_class = np.argmax(p_temp) + 1  # 得到样本i分到的类
        if p_class != X[i][2]:
            error_rate += 1
    return error_rate / num


# 单次试验求不同准则下的分类误差
def repeated_trials(mean, cov, P1, P2):
    # 根据mean，cov，P1,P2生成数据集X1,X2
    # 通过不同规则得到不同分类错误率并返回
    # 生成N=1000的数据集
    X1 = Generate_DataSet(mean, cov, P1)  # 生成1000个样本，每行[x,y,label]
    X2 = Generate_DataSet(mean, cov, P2)
    error = np.zeros((2, 2))
    # 计算似然率测试规则误差
    error_likelihood = Likelihood_Test_Rule(X1, mean, cov)
    error_likelihood_2 = Likelihood_Test_Rule(X2, mean, cov)
    error[0] = [error_likelihood, error_likelihood_2]
    # 计算最大后验概率规则误差
    error_Max_Posterior_Rule = Max_Posterior_Rule(X1, mean, cov, P1)
    error_Max_Posterior_Rule_2 = Max_Posterior_Rule(X2, mean, cov, P2)
    error[1] = [error_Max_Posterior_Rule, error_Max_Posterior_Rule_2]

    return error


# 欧几里得距离
def dis(x1, x2):
    return (x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2


# 高斯密度函数
def gauss_density_function(x, X, h):
    n = len(X)
    sum = 0
    for i in range(n):
        sum += np.exp(-dis(x, X[i]) / (2 * h * h))
    sum /= np.sqrt(2 * np.pi) * h
    p = 1 / n * sum
    return p


# 高斯核函数估计方法，交叉验证寻找最佳h值
def kfold_Likelihood_Test_Rule(X, h):
    k = 5
    kf = KFold(n_splits=k, shuffle=True)  # 5折验证
    avg_error_rate = 0
    # 对于每次交叉验证
    for train_index, test_index in kf.split(X):  # 训练集里面的编号，测试集里面的编号
        X_train = []
        X_test = []
        for i in train_index:
            X_train.append(X[i])
        for j in test_index:
            X_test.append(X[j])

        error_rate = 0
        num = len(X_test)
        for i in range(num):
            p_temp = [0, 0, 0]
            # 首先对训练集按照标签分组，在每组内计算p(x)
            for j in range(1, 4):
                X_class = []
                for x in X_train:
                    if x[2] == j:
                        X_class.append(x)
                p_temp[j - 1] = gauss_density_function(X_test[i], X_class, h)
            p_class = np.argmax(p_temp) + 1  # 得到样本i决策到的类
            if p_class != X_test[i][2]:
                error_rate += 1
        error_rate /= num
        avg_error_rate += error_rate
        print(avg_error_rate)
    avg_error_rate /= k
    return avg_error_rate


def find_best_h(mean, cov, P1, P2):
    hlist = [0.1, 0.5, 1, 1.5, 2]
    res = []
    X1 = Generate_DataSet(mean, cov, P1)  # 生成1000个样本，每行[x,y,label]
    X2 = Generate_DataSet(mean, cov, P2)
    for h in hlist:
        # 计算似然率测试规则误差
        error_likelihood = kfold_Likelihood_Test_Rule(X1, h)
        error_likelihood_2 = kfold_Likelihood_Test_Rule(X2, h)
        error = [error_likelihood, error_likelihood_2]
        res.append(error)
    print(res)


import heapq

class _KNN:
    # 计算两向量距离
    def get_dis(self, v1, v2):
        dis=(v1[0]-v2[0])**2+(v1[1]-v2[1])**2
        return dis

    # 找到最近的k个点的编号，这里使用最小堆
    def get_k_neighboors(self, dis, k):
        # 首先用字典存下来每个dis的编号
        index_dic = {}
        for i in range(len(dis)):
            index_dic[dis[i]] = i
        heapq.heapify(dis)  # 构建最小堆
        index = []
        for i in range(k):
            minx = heapq.heappop(dis)
            index.append(index_dic[minx])

        return index

    # 找到最近的k个点中出现最多的标签
    def get_max_label(self, index, y_train):
        pt_lables = [y_train[i] for i in index]
        return max(set(pt_lables), key=pt_lables.count)  # 返回众数

    # knn.fit
    def fit(self, X_train, X_test, y_train, y_test, k):
        cnt = 0
        for i in range(len(X_test)):
            dis = []
            for point in X_train:
                dis.append(self.get_dis(X_test[i], point))

            k_nb = self.get_k_neighboors(dis, k)
            result = self.get_max_label(k_nb, y_train)

            if result == y_test[i]:
                cnt += 1
        accuracy = cnt / len(X_test)
        return accuracy

    def test(self, X, y, k):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=423)
        # 误差
        error_likelihood = 1-self.fit(X_train, X_test, y_train, y_test, k)
        return error_likelihood

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
# sklearn的knn
def sklearn_knn(X, y, k):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=423)
    knn = KNN(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)
    accurary = accuracy_score(y_test, y_predict)
    return 1-accurary

if __name__ == '__main__':
    mean = np.array([[1, 1], [4, 4], [8, 1]])  # 均值数组，每个簇的各维度均值
    cov = [[2, 0], [0, 2]]  # 方差矩阵
    num = 1000  # 样本个数
    P1 = [1 / 3, 1 / 3, 1 / 3]  # 样本X1的先验概率
    P2 = [0.6, 0.3, 0.1]  # 样本X2的先验概率
    # Generate_DataSet_plot(mean, cov, P1)  # 画X1数据集散点图
    # Generate_DataSet_plot(mean, cov, P2)  # 画X2数据散点图

    # # 1.
    # # 计算十次运算的总误差
    # error_all = np.zeros((2, 2))
    # # 测试times_num次求平均
    # times_num = 10
    # for times in range(times_num):
    #     print(repeated_trials(mean,cov,P1,P2))
    #     error_all += repeated_trials(mean, cov, P1, P2)
    # # 计算平均误差
    # error_ave = error_all / times_num
    # print(error_ave)

    # 2.高斯核函数估计方法，交叉验证寻找最佳h值
    #find_best_h(mean, cov, P1, P2)

    # 3.knn寻找最佳k
    knn = _KNN()
    X1 = Generate_DataSet(mean, cov, P1)  # 生成1000个样本，每行[x,y,label]
    X2 = Generate_DataSet(mean, cov, P2)

    x_1 = []
    y_1 = []
    x_2 = []
    y_2 = []
    for line in X1:
        x_1.append(line[:2])
        y_1.append(line[2])
    for line in X2:
        x_2.append(line[:2])
        y_2.append(line[2])
    k_list = [1, 3, 5]
    for k in k_list:
        error_likelihood = knn.test(x_1, y_1, k=k)
        error_likelihood2 = knn.test(x_2, y_2, k=k)
        # error_likelihood = sklearn_knn(x_1, y_1, k=k)
        # error_likelihood2 = sklearn_knn(x_2, y_2, k=k)
        print("k=", k, ":", error_likelihood, error_likelihood2)
