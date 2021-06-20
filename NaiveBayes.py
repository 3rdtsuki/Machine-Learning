import numpy as np
import matplotlib.pyplot as plt


def make_plot(X, y, group):
    fig, ax = plt.subplots()
    ax.plot(X, y)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title("ROC curve of group " + group, fontsize=16)
    plt.show()


test_num = [0, 0, 0]


# 分层采样：每一类里选取后20%作为测试集
def read_data(X_train, X_test, y_test):
    with open("./wine.data", encoding="utf-8")as f:
        text = f.readlines()
        X = [[], [], []]
        for line in text:
            vec = line.strip().split(',')
            label = int(vec[0]) - 1
            vec = [float(i) for i in vec]
            X[label].append(vec[1:])

    k = 0.6
    for i in range(3):
        num = len(X[i])
        train_num = int(num * k)  # 该类别的训练集大小
        X_train[i] = X[i][:train_num]  # 训练集的所有向量，按照标签分成三组
        for vec in X[i][train_num:]:
            X_test.append(vec)

        test_num[i] = num - train_num  # 该类别的测试集大小
        for j in range(test_num[i]):
            y_test.append(i)


# p(xi_d|cj):类cj的均值，方差var
def Gauss_Probability(x, mean, var):
    p = 1 / (2 * np.pi * np.sqrt(var)) * np.exp(-0.5 * (x - mean) ** 2 / var)
    return p


# 朴素贝叶斯分类器，对于每类c,计算测试集中每个xi的每个属性d的条件概率p(xi_d|cj)，所有d相乘得到总的概率
def NaiveBayes(X_train, X_test, y_test):
    # 先计算每类的均值mean
    mean = []
    var = []
    for i in range(0, 3):
        mean.append(np.mean(X_train[i], axis=0))  # 每列的均值
        var.append(np.var(X_train[i], axis=0))  # 每列的方差

    num = len(X_test)
    print(num)
    error_rate = 0
    TP = np.zeros(3)  # 预测为i，实际为i
    FP = np.zeros(3)  # 预测为i，实际非i
    TN = np.zeros(3)  # 预测非i，实际非i
    FN = np.zeros(3)  # 预测非i，实际为i

    p_list=[]# 存储每个样本对3类计算出的p
    for i in range(num):  # 对每个测试集成员
        p_temp = np.zeros(3)    # 每个样本对3类计算出的p
        for j in range(0, 3):  # 对每个簇
            # 把幂运算取对数，方便加法
            p = np.log((2 * np.pi * var[j]) ** 0.5) + (np.power(X_test[i] - mean[j], 2)) / (2 * var[j])
            p = -np.sum(p) + np.log(test_num[j] / num)

            p_temp[j] = p
        p_list.append(p_temp)

        p_class = np.argmax(p_temp)  # 得到样本i决策到的类
        if p_class != y_test[i]:  #
            error_rate += 1
            FP[p_class] += 1
            FN[y_test[i]] += 1
        else:
            TP[p_class] += 1
            for j in range(3):
                if j != p_class:
                    TN[j] += 1
    for j in range(3):
        print("类%d的混淆矩阵为：\nTP=%d, FP=%d, TN=%d, FN=%d" % (j + 1, TP[j], FP[j], TN[j], FN[j]))
        print("Accuracy:", (TP[j] + TN[j]) / (TP[j] + FP[j] + TN[j] + FN[j]))
        recall = TP[j] / (TP[j] + FN[j])
        precision = TP[j] / (TP[j] + FP[j])
        print("Recall:", recall)
        F1 = 2 * recall * precision / (precision + recall)
        print("F1 score:", F1)
    accuracy = 1 - error_rate / num
    print("General Accuracy:", accuracy)
    return p_list


'''
ROC限定朴素贝叶斯分类器
threshold：画roc曲线用的阈值
group：限定的类别。如果group=i，那么p算出来只要超过阈值，那么测试点就归为i
'''


def NaiveBayes_limited(X_train, X_test, y_test, group, threshold):
    # 先计算每类的均值mean
    mean = []
    var = []
    for i in range(0, 3):
        mean.append(np.mean(X_train[i], axis=0))  # 每列的均值
        var.append(np.var(X_train[i], axis=0))  # 每列的方差

    num = len(X_test)
    TP = 0  # 预测为i，实际为i，正确
    FP = 0  # 预测为i，实际非i
    TN = 0  # 预测非i，实际非i，正确
    FN = 0  # 预测非i，实际为i

    for i in range(num):  # 对每个测试集成员
        p_temp = np.zeros(3)
        j = group  # 只测试限定的簇
        # 把幂运算取对数，方便加法
        p = np.log((2 * np.pi * var[j]) ** 0.5) + (np.power(X_test[i] - mean[j], 2)) / (2 * var[j])
        p = -np.sum(p) + np.log(test_num[j] / num)

        p_temp[j] = p
        if p >= threshold:  # 大于阈值
            p_class = j
        else:  # 小于阈值，不关心分到哪类，设为-1
            p_class = -1

        if j != y_test[i] and p_class == j:
            FP += 1
        elif j != y_test[i] and p_class != j:
            TN += 1
        elif j == y_test[i] and p_class == j:
            TP += 1
        elif j == y_test[i] and p_class != j:
            FN += 1
    #print("类%d的混淆矩阵为：\nTP=%d, FP=%d, TN=%d, FN=%d" % (group + 1, TP, FP, TN, FN))
    return FP / (FP + TN), TP / (TP + FN)  # FPR, TPR


# ROC曲线：横轴FPR，纵轴TPR，调整归为i类的p的阈值大小，因此参数需要i和threshold
def ROC(X_train, X_test, y_test,p_list):
    p_list=np.array(p_list).T

    for j in range(0, 3):
        threshold =sorted(p_list[j])
        #print(threshold)
        fpr_list = []
        tpr_list = []
        for i in threshold:
            FPR, TPR = NaiveBayes_limited(X_train, X_test, y_test, group=j, threshold=i)
            fpr_list.append(FPR)
            tpr_list.append(TPR)
        # print("fpr_list:", fpr_list)
        # print("tpr_list:", tpr_list)
        AUC(fpr_list, tpr_list)
        make_plot(fpr_list, tpr_list, str(j + 1))


'''
AUC值=ROC曲线（直方图形）下的面积
'''


def AUC(fpr_list, tpr_list):
    s = 0
    n = len(fpr_list)
    for i in range(n - 1):
        s += (fpr_list[i] - fpr_list[i+1]) * tpr_list[i+1]
    print(s)
    return s


if __name__ == "__main__":
    X_train = [[], [], []]  # 3个标签
    X_test = []
    y_test = []
    read_data(X_train, X_test, y_test)
    # 1.计算分类准确率 2.混淆矩阵，精度，召回率，F值
    p_list=NaiveBayes(X_train, X_test, y_test)
    # 3.ROC曲线和AUC值
    ROC(X_train, X_test, y_test,p_list)
