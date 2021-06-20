import csv
import matplotlib.pyplot as plt
import numpy as np
import heapq

from mpl_toolkits.mplot3d import Axes3D


# 绘制三维散点图，vec：每个元素都是[x,y,z]
def make_scatter_plot(vec, label, title):
    fig = plt.figure()
    ax = Axes3D(fig)

    n = len(X)
    color_list = ['blue', 'red', 'green']
    for i in range(n):
        # 共有三类label，所以x,y,z都分成三类
        x, y, z = [[], [], []], [[], [], []], [[], [], []]
        x[label[i]].append(vec[i][0])
        y[label[i]].append(vec[i][1])
        z[label[i]].append(vec[i][2])
        plt.scatter(x[label[i]], y[label[i]], z[label[i]], c=color_list[label[i]], label='类别' + str(label[i]))

    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

    ax.set_title(title)
    plt.show()


# 根据三个中心，利用高斯分布生成3n个点
def gen_data(n):
    mean = [[1, 1, 1], [5, 2, 5], [3, 8, 6]]
    # 协方差矩阵
    cov = [[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]]

    X = np.random.multivariate_normal(mean[0], cov, n)  # 生成n个三维向量
    X = np.append(X, np.random.multivariate_normal(mean[1], cov, n), axis=0)
    X = np.append(X, np.random.multivariate_normal(mean[2], cov, n), axis=0)

    X = [list(i) for i in X]
    # 每个向量加上一维作为label
    for i in range(n):
        X[i].append(0)
    for i in range(n, 2 * n):
        X[i].append(1)
    for i in range(2 * n, 3 * n):
        X[i].append(2)

    n = 3 * n
    for i in range(n):
        for j in range(3):
            if X[i][j] < 0:
                X[i][j] = -X[i][j]  # 不能有负数！
            X[i][j] = round(X[i][j], 1)

    write_into_csv('./training.csv', X)


def read_from_csv(path):
    X = []
    y = []
    csv.field_size_limit(500 * 1024 * 1024)
    with open(path, 'r', encoding="utf-8", newline="")as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            vec = [float(line[i]) for i in range(3)]
            X.append(vec)
            y.append(int(line[3]))
    return X, y  # items是一个二维数组


def write_into_csv(path, items):
    with open(path, 'w', encoding="utf-8", newline="")as fp:
        writer = csv.writer(fp)
        writer.writerows(items)


# 两个点的欧氏距离,ord=2
def euler_dis(vec1: list, vec2: list) -> float:
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.linalg.norm(vec1 - vec2, ord=2)


# a,b按从小到大排列
def sort(a, b):
    if a > b:
        t = a
        a = b
        b = t
    return a, b


class Node(object):  # 每个簇
    def __init__(self, id, left=None, right=None, count=1):
        """
        :param id: 编号
        :param left: 左儿子
        :param right: 右儿子
        :param count: 叶节点数
        """
        self.id = id
        self.left = left
        self.right = right
        self.count = count  # 包含的叶节点数量


class Hierarchical(object):
    def __init__(self, X, k=3):
        self.k = k  # 最终聚为几类
        self.X = X  # 样本，每行是一个向量
        self.nodes = {}  # 存储所有簇{id1:Node1}
        self.init_nodes()  # 初始化self.nodes
        print("init_nodes finish!")

        self.D = {}  # 距离字典{(id1,id2):dis_1_2}
        self.init_D()  # 初始化self.D
        print("init_D finish!")

    def init_nodes(self):
        nodes = [Node(id=i) for i, v in enumerate(self.X)]
        for i in range(len(nodes)):
            self.nodes[i] = nodes[i]

    def init_D(self):
        for id1 in self.nodes.keys():
            for id2 in self.nodes.keys():
                if id1 >= id2:
                    continue
                self.D[(id1, id2)] = euler_dis(self.X[id1], self.X[id2])

    def update_D(self, aid=0, bid=0, merged_id=0, method="min"):  # 更新距离字典
        self.D.pop(sort(aid, bid))
        # 首先加入一行一列，合并后的簇与其他簇的距离
        for id1 in self.nodes.keys():
            if id1 == aid or id1 == bid or id1 == merged_id:  # 合并前的id，没有用，不去计算
                continue
            dis_a = self.D[sort(aid, id1)]
            dis_b = self.D[sort(bid, id1)]
            if method == "min":
                new_dis = min(dis_a, dis_b)  # 合并后簇与i的距离是min{a-i,b-i}
            elif method == "max":
                new_dis = max(dis_a, dis_b)
            elif method == "avg":  # 加权平均值
                cnt_a, cnt_b = self.nodes[aid].count, self.nodes[bid].count
                new_dis = (dis_a * cnt_a + dis_b * cnt_b) / (cnt_a + cnt_b)
            else:
                raise Exception("Wrong method!")
            self.D.pop(sort(aid, id1))  # 删掉与ab有关的距离
            self.D.pop(sort(bid, id1))
            self.D[sort(merged_id, id1)] = new_dis  # 设置新合并的节点与i的距离

    def update_nodes(self, aid, bid):
        # 删除aid和bid对应的node
        self.nodes.pop(aid)
        self.nodes.pop(bid)

    # 返回最近两簇的id
    def find_closest(self):
        pairs = heapq.nsmallest(1, self.D, lambda x: self.D[x])  # 利用最小堆返回value最小的key:(aid,bid)
        aid, bid = pairs[0]
        return aid, bid

    # 层次聚类的入口，三种方法：min,max,avg
    def entrance(self, method="min"):
        n = len(self.nodes)
        merged_id = n
        while n > self.k:
            aid, bid = self.find_closest()  # 最近两个簇的id
            # 生成新的簇，编号为n+1
            merged_node = Node(id=merged_id, left=self.nodes[aid], right=self.nodes[bid],
                               count=self.nodes[aid].count + self.nodes[bid].count)
            # 在nodes中加入新合并的簇
            self.nodes[merged_id] = merged_node
            self.update_D(aid, bid, merged_id, method=method)  # 更新距离字典D
            self.update_nodes(aid, bid)  # 更新nodes
            merged_id += 1
            n = len(self.nodes)
            print(n)
        self.bfs()  # 对最后的三个簇用bfs搜索叶节点，为了作图
        self.write_leaves_into_csv(method=method)  # 将聚类结果写入文件

    # bfs搜索叶节点
    def bfs(self):
        self.classified_leaves = []
        for i in self.nodes.keys():
            sons = []
            q = [self.nodes[i]]
            while len(q):
                top = q[-1]
                q.pop()
                if top.left is None and top.right is None:
                    sons.append(top.id)
                if top.left:
                    q.append(top.left)
                if top.right:
                    q.append(top.right)
            self.classified_leaves.append(sons)
            # print(sons)

    def write_leaves_into_csv(self, method):
        result = []  # x,y,z,label
        for i in range(3):
            for id in self.classified_leaves[i]:
                result.append(self.X[id] + [i])
        path = "./classified_result_" + method + ".csv"
        write_into_csv(path, result)


if __name__ == "__main__":
    # path = r'./iris_training.csv' # 鸢尾花数据集

    # gen_data(667) # 造数据

    method = "min"
    # path = r'./training.csv'
    # X, y = read_from_csv(path)
    # Hier = Hierarchical(X=X, k=3)
    # Hier.entrance(method)

    result_path = "./classified_result_" + method + ".csv"
    X, y = read_from_csv(result_path)
    make_scatter_plot(X, y, title="Single Linkage")
    # make_scatter_plot(X, y, title="Complete Linkage")
    # make_scatter_plot(X, y, title="Average Linkage")
