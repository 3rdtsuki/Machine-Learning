"""
只含离散特征的决策树
"""
import csv
import math

color = {'浅白': 0, '乌黑': 1, '青绿': 2}
voice = {'沉闷': 0, '浊响': 1, '清脆': 2}
root = {'蜷缩': 0, '稍蜷': 1, '硬挺': 2}
texture = {'清晰': 0, '稍糊': 1, '模糊': 2}
judge = {'否': 0, '是': 1}


def read_from_csv(path):
    X = []
    y = []
    csv.field_size_limit(500 * 1024 * 1024)
    with open(path, 'r', encoding="gbk", newline="")as csvfile:
        reader = csv.reader(csvfile)
        cnt = 0
        for line in reader:
            if cnt == 0:
                cnt = 1
                continue
            vec = []
            vec.append(color[line[1]])
            vec.append(root[line[2]])
            vec.append(voice[line[3]])
            vec.append(texture[line[4]])
            X.append(vec)
            y.append(judge[line[5]])
    return X, y  # items是一个二维数组


# 决策树的每个节点类
class Node(object):
    def __init__(self, id=0, depth=0, value=None):
        self.feature = None  # 该节点由什么特征来划分
        self.sons = []  # 该节点的子节点们
        self.label = None  # 该节点最终被预测为什么标签
        self.depth = depth  # 节点的深度
        self.id = id
        self.value = value  # 该节点是由父节点的哪种特征值划分来的


# 决策树类
class DecisionTree(object):
    def __init__(self, X, y, root, method="ID3"):
        """
        :param X: 样本
        :param y: labels
        :param root: 根节点
        :param method: 方法：ID3,C4.5,CART
        """
        self.X = X
        self.y = y
        self.n = len(self.X)
        self.root = root
        self.method = method
        self.cnt = 0

    # 原始熵
    def entropy(self, index_list):
        n = len(index_list)
        # 原始熵
        e = 0
        cnt0 = 0  # label=0的个数
        for i in index_list:
            if self.y[i] == 0:
                cnt0 += 1
        p0 = cnt0 / n
        p1 = 1 - p0
        if p0 != 0:
            e += p0 * math.log(p0, 2)  # 注意底数在后面
        if p1 != 0:
            e += p1 * math.log(p1, 2)
        e = -e
        return e

    # 条件熵：按照feature分类后计算原始熵的加权平均。[5/14(2/5log2/5+3/5log3/5)+...]=[cnt_class/n*temp_e]
    def condition_entropy(self, feature, index_list):
        n = len(index_list)
        condition_e = 0
        for j in range(3):  # 对于feature的每类
            class_list = []
            for i in index_list:
                if self.X[i][feature] == j:
                    class_list.append(i)
            if len(class_list) == 0:  # 这个类没有样本，不用计算
                continue
            condition_e += len(class_list) / n * self.entropy(class_list)  # 子类的原始熵相加
        # 因为是子类的原始熵相加，所以条件熵不用再加负号了
        return condition_e

        # 数据集D关于特征A的值的熵

    # 数据集D关于特征A的值的熵
    def HAD_entropy(self, feature, index_list):
        n = len(index_list)
        had = 0
        for j in range(3):
            cnt_class = 0
            for i in index_list:
                if self.X[i][feature] == j:
                    cnt_class += 1
            p = cnt_class / n
            if p != 0:
                had += p * math.log(p, 2)
        had = -had
        return had

    # 基尼指数，1-sum{(Ci/n)^2}这里C只有两类0和1，所以k=2
    def Gini(self, index_list):
        n = len(index_list)
        cnt0 = 0
        for i in index_list:
            if self.y[i] == 0:
                cnt0 += 1
        g = 1 - (cnt0 / n) ** 2 - (1 - cnt0 / n) ** 2
        return g

    # 将集合D按照A属性划分后的基尼指数
    def Gini_D_A(self, feature, index_list):
        n = len(index_list)
        g = 0
        for j in range(3):
            cnt_class = 0
            index_class = []
            for i in index_list:
                if self.X[i][feature] == j:
                    cnt_class += 1
                    index_class.append(i)
            p = cnt_class / n
            if cnt_class:
                g += p * self.Gini(index_class)
        return g

    # 计算信息增益：前后熵之差
    def delta_entropy(self, feature, index_list):
        n = len(index_list)
        if n == 0:
            return 0
        # 原始熵
        e = self.entropy(index_list)
        # 条件熵
        condition_e = self.condition_entropy(feature, index_list)
        # 信息增益
        delta_e = e - condition_e
        # H_A(D)
        had_e = self.HAD_entropy(feature, index_list)
        # 基尼指数
        delta_gini = self.Gini(index_list) - self.Gini_D_A(feature, index_list)

        # ID3找熵变化最大的，C4.5找信息增益比最大的，CART找基尼指数变化最大的
        if self.method == "ID3":
            return delta_e
        elif self.method == "C4.5":
            return 0 if had_e == 0 else delta_e / had_e
        elif self.method == "CART":
            return delta_gini
        else:
            raise Exception("method error!")

    # 在index_list范围内寻找特征
    def find_best_feature(self, index_list):
        max_delta_e = 0
        res = 0
        for i in range(4):  # 4个特征
            # 信息增益
            delta_e = self.delta_entropy(feature=i, index_list=index_list)
            if delta_e > max_delta_e:  # 找最大的熵变化量
                max_delta_e = delta_e
                res = i
        return res

    # 以当前节点node构建决策树，该节点包含index_list里面的样本
    def build_tree(self, now_node, index_list):
        # 终止条件
        # 如果列表中每个样本的lable都相同，结束递归
        all_labels_same = True
        now_label = self.y[index_list[0]]
        for i in index_list:
            if self.y[i] != now_label:
                all_labels_same = False
                break
        if all_labels_same:
            now_node.label = now_label
            return
        """
        高级要求：剪枝，如果深度达到k也结束递归。如果当前集合里面大多数样本标签为0，则设定当前节点标签为0
        """
        # if now_node.depth == 2:
        #     cnt0 = 0
        #     for i in index_list:
        #         if self.y[i] == 0:
        #             cnt0 += 1
        #     if cnt0 >= len(index_list) / 2:
        #         now_node.label = 0
        #     else:
        #         now_node.label = 1
        #     return

        # 选择最佳特征，使得熵下降最大
        now_node.feature = self.find_best_feature(index_list)

        # 根据特征的值分成3堆
        group_list = [[], [], []]
        for j in range(3):
            for i in index_list:
                if self.X[i][now_node.feature] == j:  # 如果特征的值为j
                    group_list[j].append(i)
            if len(group_list[j]):  # 如果这个特征值没有样本点满足，就不创建节点了
                # 每个特征值创建一个节点
                self.cnt += 1
                son_node = Node(id=self.cnt, depth=now_node.depth + 1, value=j)
                now_node.sons.append(son_node)  # 建立父子节点之间的连接
                self.build_tree(now_node=son_node, index_list=group_list[j])  # 分治处理每个堆

    # 输入测试集，对每个测试点进行分类预测
    def fit(self, test_X, test_y):
        error = 0
        n = len(test_y)
        for i in range(n):
            now_node = self.root

            while len(now_node.sons):
                f = now_node.feature
                now_node = now_node.sons[test_X[i][f]]  # 根据x的f特征的值决定走向哪个子节点
            print(now_node.label, test_y[i])
            if now_node.label != test_y[i]:
                error += 1
        accurate = 1 - error / n
        print("accurate:", accurate)
        return accurate

    # 通过bfs输出整棵树
    def bfs(self):
        attr = ['色泽', '根蒂', '敲声', '纹理']
        feature_dic = {'色泽': ['浅白', '乌黑', '青绿'], '敲声': ['沉闷', '浊响', '清脆'],
                       '根蒂': ['蜷缩', '稍蜷', '硬挺'], '纹理': ['清晰', '稍糊', '模糊']}
        no_yes = ['否', '是']
        nodes = [self.root]
        while len(nodes):
            top = nodes[-1]
            print(top.id, end=" ")
            if top.feature != None:
                print(attr[top.feature], end="->")
                for i in top.sons:
                    print("[son:%d, value:%s]" % (i.id, feature_dic[attr[top.feature]][i.value]), end=",")
            if top.label != None:  # 该点是叶节点
                print(no_yes[top.label], end="")
            print()
            nodes.pop()
            nodes += top.sons


if __name__ == "__main__":
    train_path = "./Watermelon-train1.csv"
    test_path = "./Watermelon-test1.csv"
    train_X, train_y = read_from_csv(train_path)
    test_X, test_y = read_from_csv(test_path)

    index_list = [i for i in range(len(train_y))]
    root = Node()
    method = ["ID3", "C4.5", "CART"]
    DT = DecisionTree(train_X, train_y, root, method=method[0])
    DT.build_tree(now_node=root, index_list=index_list)

    # DT.fit(test_X, test_y)
    DT.bfs()
