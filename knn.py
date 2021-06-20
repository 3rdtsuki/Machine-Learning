# kNN实现手写数字识别
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import heapq
import matplotlib.pyplot as plt

# 读数据，数据格式为256个像素，后面跟着10位标签
path = "./semeion.data"
vecs = []
labels = []
with open(path, 'r', encoding="gbk", newline="")as f:
    text = f.readlines()
    for line in text:
        vec = line.strip().split(' ')
        pic_vec = vec[:-10]  # 坐标部分
        label_vec = vec[-10:]  # 标签部分
        label = 0  # 标签
        for i in range(len(label_vec)):
            if label_vec[i] == '1':
                label = (10 - i) % 10
                break
        pic_vec = [float(i) for i in pic_vec]
        vecs.append(pic_vec)
        labels.append(label)


# 计算两向量距离
def get_dis(v1, v2):
    dis = 0
    for i in range(len(v1)):
        if v1[i] != v2[i]:
            dis += 1
    return dis


# 找到最近的k个点的编号，这里使用最小堆
def get_k_neighboors(dis, k):
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
def get_max_label(index, y_train):
    pt_lables = [y_train[i] for i in index]
    return max(set(pt_lables), key=pt_lables.count)  # 返回众数

# 自己实现的knn
def knn(X_train, X_test, y_train, y_test, k):
    cnt = 0
    for i in range(len(X_test)):
        dis = []
        for point in X_train:
            dis.append(get_dis(X_test[i], point))

        k_nb = get_k_neighboors(dis, k)
        result = get_max_label(k_nb, y_train)

        # tuples = sorted(enumerate(dis), key=lambda x: x[1])[:k]
        # tuples=[i[0]for i in tuples]
        # result=get_max_label(tuples,y_train)

        if result == y_test[i]:
            cnt += 1
    accuracy = cnt / len(X_test)
    return accuracy


# 中级要求：与sklearn的knn进行对比
def sklearn_knn(X_train, X_test, y_train, y_test, k):
    knn = KNN(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)
    accurary = accuracy_score(y_test, y_predict)
    return accurary


# 提高要求：绘制交叉验证结果，y轴为分类精度，留一法：n-1折验证
def kfold(X, y, k):
    kf = KFold(n_splits=5, shuffle=True)  # 5折验证
    sum = 0
    cnt = 0
    for train_index, test_index in kf.split(X):  # 训练集里面的编号，测试集里面的编号
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for i in train_index:
            X_train.append(X[i])
            y_train.append(y[i])
        for j in test_index:
            X_test.append(X[j])
            y_test.append(y[j])
        # result=knn(X_train, X_test, y_train, y_test,k)
        result = sklearn_knn(X_train, X_test, y_train, y_test, k)
        sum += result
        cnt += 1
    avg = sum / cnt
    # print("k fold accuracy:",avg)
    return avg

# 绘制k-错误率折线图
def make_plot_pic(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('k')
    ax.set_ylabel('error rate')
    plt.xticks([i for i in range(1, 21)])
    plt.show()


if __name__ == "__main__":
    k_list = [1,3,5]
    # k_list=[i for i in range(1,21)]
    result_list = []
    # 划分出训练集、测试集的输入x，训练集、测试集的值y
    X_train, X_test, y_train, y_test = train_test_split(vecs, labels, test_size=0.2, random_state=423)
    for k in k_list:
        print("k=", k)
        result=knn(X_train, X_test, y_train, y_test, k=k)
        #result=sklearn_knn(X_train, X_test, y_train, y_test, k=k)

    #     result=kfold(vecs, labels, k=k)
        print(result)
    #     result_list.append(1-result)
    # make_plot_pic(k_list,result_list)

