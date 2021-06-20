import csv
import random
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_from_csv(path):
    items = []
    csv.field_size_limit(500 * 1024 * 1024)
    with open(path, 'r', encoding='utf-8')as f:
        reader = csv.reader(f)
        cnt = 0
        for line in reader:
            if cnt:
                line = [float(i) for i in line]
                items.append(line)
            cnt += 1
    return items


# 折线图
def make_plot_pic(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # plt.xticks(x)
    plt.show()


# 最小二乘法
def ordinary_least_squares(X, y):
    sumxy = sumx = sumy = sumx2 = 0
    n = len(X)
    for i in range(n):
        sumx += X[i]
        sumx2 += X[i] ** 2
        sumy += y[i]
        sumxy += X[i] * y[i]
    a = (sumxy - sumx * sumy / n) / (sumx2 - sumx ** 2 / n)
    b = sumy / n - a * sumx / n
    return a, b


# 一维线性回归的MSE
def linear_MSE(X, y, a, b):
    mse = 0
    n = len(X)
    for i in range(n):
        mse += (y[i] - a * X[i] - b) ** 2
    mse /= n
    return mse


# 多元回归的MSE
def MSE(X, y, theta):
    mse = 0
    n = len(X)
    for i in range(n):
        h = 0
        for j in range(12):
            h += theta[j] * X[i][j]
        mse += (y[i] - h) ** 2
    mse /= 2 * n
    return mse


# 将x标准化，使均值为0，落在[-1,1]
def Normalization_fun(x):
    x = (x - np.mean(x, 0)) / (np.max(x, 0) - np.min(x, 0))
    return x


# 批量梯度下降Batch Gradient Descent
def BGD(X, y, learning_rate):
    m = X.shape[1]  # X的列数
    n = X.shape[0]  # X的行数
    theta = np.zeros((m, 1))  # 11个特征的系数+一个常数！
    mse = [0xffff, 0xfff]
    cnt = 0
    while np.abs(mse[-1] - mse[-2]) > 0.00001:
        cnt += 1
        w_temps = np.zeros((theta.shape[0], 1))
        for j in range(m):
            w_temps[j] = theta[j] + learning_rate * np.dot((y - np.dot(X, theta)).T, X[:, j]) / n
        theta = w_temps
        # 计算一下MSE
        new_mse = np.linalg.norm(np.dot(X, theta) - y) ** 2 / n
        mse.append(new_mse)
        # print(new_mse)
    # 迭代次数-mse图像
    x = [i for i in range(1, cnt + 1)]
    make_plot_pic(x, mse[2:])
    return theta, mse[-1]


# 随机梯度下降Stochastic Gradient Descent
def SGD(X, y, learning_rate):
    m = X.shape[1]  # X的列数
    n = X.shape[0]
    theta = np.zeros((m, 1))
    mse = [0xffff, 0xfff]
    while np.abs(mse[-1] - mse[-2]) > 0.0001:
        # 每次随机选择一个点i
        # for i in random.sample(range(n), n):
        i = random.randint(0, n)
        # xi对于这些theta的预测值
        h = np.dot(X[i][:], theta)
        for j in range(m):
            theta[j] += learning_rate * (y[i] - h) * X[i][j]
        new_mse = np.linalg.norm(np.dot(X, theta) - y) ** 2 / n
        mse.append(new_mse)
        print(new_mse)
    print(len(mse[2:]))
    return theta


# 岭回归
def RidgeRegression(X, y, learning_rate, _lambda):
    m = X.shape[1]  # X的列数
    n = X.shape[0]  # X的行数
    theta = np.zeros((m, 1))  # 11个特征的系数+一个常数！
    mse = [0xffff, 0xfff]
    cnt = 0
    while np.abs(mse[-1] - mse[-2]) > 0.0001:
        cnt += 1
        h = np.dot(X, theta)
        h = (y - h).T
        for j in range(m):
            theta[j] = theta[j] + learning_rate * np.dot(h, X[:, j]) / n + _lambda * theta[j]
        # 计算一下MSE
        new_mse = np.linalg.norm(np.dot(X, theta) - y) ** 2 / n + _lambda * sum([i ** 2 for i in theta])
        mse.append(new_mse)
        print(new_mse)
    print(len(mse[2:]))
    return theta


def split(a, b, n):
    step = (b - a) / n
    x = [a + step * i for i in range(n)]
    return x


import time

if __name__ == '__main__':
    # # 1.1 最小二乘法
    # path = './dataset_regression.csv'
    # table = read_from_csv(path)
    # X = []
    # y = []
    # for line in table:
    #     X.append(line[1])
    #     y.append(line[2])
    # print(X,y)
    # a, b = ordinary_least_squares(X, y)
    # print(a, b)
    #
    # # 训练误差
    # train_MSE = linear_MSE(X, y, a, b)
    # print("训练误差为：",train_MSE)
    #
    # # 新增5个点，加噪声，测试误差
    # X_test = [-1.25, -0.25, 0.25, 1.25, 1.75]
    # y_test = [a * i + b for i in X_test]
    # for i in range(len(y_test)):
    #     y_test[i] += random.gauss(mu=0, sigma=0.12)  # 高斯噪声的均值和方差
    # test_MSE=linear_MSE(X_test,y_test,a,b)
    # print("测试误差为：",test_MSE)
    #
    # # 散点图
    # fig, ax = plt.subplots()
    # ax.scatter(X+X_test, y+y_test)
    # ax.plot([-2,2],[-2*a+b,2*a+b],color='red')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # plt.show()

    # 1.2 BGD SGD
    path = "./winequality-white.csv"
    table = pd.read_csv(path)
    X = np.array(table.iloc[:, :-1])  # 除了最后一列的所有列
    X = Normalization_fun(X)  # 将x标准化
    y = np.array(table.iloc[:, -1]).reshape(X.shape[0], 1)  # y是n行1列的列向量
    # 注意11个特征需要12个参数theta
    # 因为theta有常数项，所以X的每一行都需要增加一个维度，值恒为1
    X = np.vstack((np.ones(X.shape[0]), X.T)).T  # 将x先转置，上面加一行，然后再转置，相当于左边加了一列
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=423)
    # 批量梯度下降
    # theta,mse = BGD(X_train, y_train, learning_rate=0.1)
    # print(theta)
    # train_mse=np.linalg.norm(np.dot(X_train, theta) - y_train) ** 2 /X_train.shape[0]
    # test_mse=np.linalg.norm(np.dot(X_test, theta) - y_test) ** 2 /X_test.shape[0]
    # print("train_mse:",train_mse)
    # print("test_mse:",test_mse)
    # 随机梯度下降
    # theta = SGD(X_train, y_train, learning_rate=0.1)

    # 2.尝试不同学习率
    # mses = []
    # rates = [1, 0.1, 0.01, 0.001]
    # for rate in rates:
    #     start = time.time()
    #     theta, mse = BGD(X_train, y_train, learning_rate=rate)
    #     end = time.time()
    #     print("time:", end - start)
    #     mses.append(mse)
    # print(mses)
    # make_plot_pic(rates,mses)

    # 3.岭回归 RidgeRegression
    theta = RidgeRegression(X_train, y_train, learning_rate=0.1, _lambda=0.001)
    print(theta)
    train_mse = np.linalg.norm(np.dot(X_train, theta) - y_train) ** 2 / X_train.shape[0]
    test_mse = np.linalg.norm(np.dot(X_test, theta) - y_test) ** 2 / X_test.shape[0]
    print("train_mse:", train_mse)
    print("test_mse:", test_mse)
