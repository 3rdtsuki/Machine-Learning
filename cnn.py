import numpy as np
from functools import reduce
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


# 可视化每个 epoch 的训练误差loss、训练精度accuracy、测试误差和测试精度
def make_plot(x, y, ylabel, xlabel="epoch"):
    fig, ax = plt.subplots()
    ax.plot(x, y, marker=None)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(xlabel + "-" + ylabel + " curve")
    plt.show()


# 激活函数
class Sigmoid(object):
    def __init__(self, input_shape):
        self.output_shape = input_shape
        self.output = np.zeros(self.output_shape)
        self.delta = np.zeros(input_shape)
        self.x = np.zeros(input_shape)

    def forward(self, x):
        self.x = x
        self.output = 1 / (1 + np.exp(-self.x))
        return self.output

    def backward(self, delta):
        self.delta = delta * self.output * (1 - self.output)
        return self.delta


class Relu(object):
    def __init__(self, input_shape):
        self.delta = np.zeros(input_shape)
        self.x = np.zeros(input_shape)
        self.output_shape = input_shape

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)  # relu就是把数组中负数都变成0

    def backward(self, delta):
        self.delta = delta
        self.delta[self.x < 0] = 0  # x中小于0元素对应的项的delta=0
        return self.delta


class Softmax(object):
    def __init__(self, input_shape):
        self.softmax = np.zeros(input_shape)
        self.delta = np.zeros(input_shape)
        self.batch_size = input_shape[0]

    # 计算评估值loss
    def loss(self, output, labels):
        self.labels = labels
        self.predict(output)
        loss = 0
        for i in range(self.batch_size):
            loss += np.log(np.sum(np.exp(output[i]))) - output[i][
                self.labels[i]]  # 设labels[i]=2，那么如果output[i][2]=1，则loss=0，否则=1
        return loss

    # 将全连接层的线性输出转为概率分布
    def predict(self, output):
        self.softmax = np.zeros(output.shape)
        exp_output = np.zeros(output.shape)
        for i in range(self.batch_size):
            output[i:, ] -= np.max(output[i])  # 先减去最大值，否则计算exp时会越界
            exp_output[i] = np.exp(output[i])
            self.softmax[i] = exp_output[i] / np.sum(exp_output[i])
        return self.softmax

    # 生成反向传播的delta
    def backward(self):
        self.delta = self.softmax.copy()
        for i in range(self.batch_size):
            self.delta[i][self.labels[i]] -= 1
        return self.delta


# 卷积层
class ConvolutionLayer(object):
    def __init__(self, input_shape, output_channels, kernel_size=3, stride=1):  # 3*3的卷积核
        self.input_shape = input_shape  # batch_size*w*h*通道数
        self.output_channels = output_channels  # 卷积核数
        self.input_channels = input_shape[-1]  # 输入图片的通道数，注意如果输入是3通道的，那么卷积核也应该是三通道的（3倍的参数）
        self.batch_size = input_shape[0]
        self.stride = stride
        self.kernel_size = kernel_size

        weights_scale = np.sqrt(reduce(lambda x, y: x * y, input_shape) / self.output_channels)
        # 卷积核大小*图片通道数*卷积核数
        self.weights = np.random.standard_normal(
            (kernel_size, kernel_size, self.input_channels, self.output_channels)) / weights_scale
        # b的数量=卷积核数
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale
        # 输出形状：batch*h*w*卷积核数
        self.delta = np.zeros((self.batch_size, (input_shape[1] - kernel_size + 1) // stride,
                               (input_shape[2] - kernel_size + 1) // stride, self.output_channels))
        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)
        self.output_shape = self.delta.shape

    # 把图片中每个卷积核对应的子矩阵转成行向量，拼成一个新的矩阵
    def im2col(self, img):
        col_img = []
        for i in range(0, img.shape[1] - self.kernel_size + 1, self.stride):
            for j in range(0, img.shape[2] - self.kernel_size + 1, self.stride):
                col = img[:, i:i + self.kernel_size, j:j + self.kernel_size, :].reshape(-1)  # 图片是四维的，只截取2,3维长和宽
                col_img.append(col)
        col_img = np.array(col_img)
        return col_img

    def forward(self, input_array):
        # 把卷积核转为列向量，列数为卷积核数，行数是通道数*卷积核大小
        col_weights = self.weights.reshape((-1, self.output_channels))
        output = np.zeros(self.output_shape)
        self.col_img = []

        for i in range(self.batch_size):
            img = input_array[i][np.newaxis, :]  # 在最前面增加一个维度（即对于一张图batchsize=1）
            # 对于每张图，都通过im2col变形为：子矩阵数*卷积核hw
            col_img_i = self.im2col(img)
            self.col_img.append(col_img_i)
            # 变形后的img和列卷积核相乘，结果转为原来形状。output=conv(img,kernel)
            output[i] = np.reshape(np.dot(col_img_i, col_weights) + self.bias, self.delta[0].shape)
        self.col_img = np.array(self.col_img)
        return output

    def backward(self, delta):
        self.delta = delta
        # 将delta转为batch*输出的hw*卷积核数的形状，而hw=子矩阵数
        col_delta = np.reshape(delta, (self.batch_size, -1, self.output_channels))
        # 分别计算w和b的梯度
        for i in range(self.batch_size):
            self.w_grad += np.dot(self.col_img[i].T, col_delta[i]).reshape(self.weights.shape)  # 输入与delta做卷积
        self.b_grad += np.sum(col_delta, axis=(0, 1))  # 每列的和，拼成一个向量

        # 计算向前传播的next_delta=conv(delta,np.rot90(kernel,2))，为了让next_delta和img形状相同，需要将delta用pad扩充，使得结果符合形状
        pad_delta = np.pad(array=self.delta,
                           pad_width=((0, 0), (self.kernel_size - 1, self.kernel_size - 1),
                                      (self.kernel_size - 1, self.kernel_size - 1), (0, 0)),
                           mode='constant',
                           constant_values=0)
        col_pad_delta = np.array([self.im2col(pad_delta[i][np.newaxis, :]) for i in range(self.batch_size)])
        # 将卷积核翻转180度
        flip_weights = np.flipud(np.fliplr(self.weights))  # fliplr左右翻转，flipud上下翻转
        flip_weights = flip_weights.swapaxes(2, 3)  # 交换2,3维
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        # col_rot_weights = np.rot90(self.weights, 2).reshape((-1, self.input_channels))
        next_delta = np.dot(col_pad_delta, col_flip_weights)
        next_delta = np.reshape(next_delta, self.input_shape)
        return next_delta

    def update(self, learning_rate):
        self.weights -= learning_rate * self.w_grad
        self.bias -= learning_rate * self.b_grad

        self.w_grad = np.zeros(self.w_grad.shape)
        self.b_grad = np.zeros(self.b_grad.shape)


# 池化层
class PoolingLayer(object):
    def __init__(self, input_shape, kernel_size=2, stride=2):
        self.input_shape = input_shape
        self.batch_size = input_shape[0]
        self.kernel_size = kernel_size
        self.stride = stride

        self.output_channels = input_shape[-1]
        # 注意如果除不开的话，输出时直接舍掉后面的行和列
        self.output_shape = [self.batch_size, input_shape[1] // stride, input_shape[2] // stride, self.output_channels]
        # 存储池化位置
        self.index = np.zeros(self.input_shape)

    # 输入(batch_size,h,w,kernel_num)
    def forward(self, input_array):
        output = np.zeros(self.output_shape)
        for b in range(self.batch_size):
            for k in range(self.output_channels):
                for i in range(0, input_array.shape[1], self.stride):
                    if i + self.kernel_size > input_array.shape[1]:
                        break
                    for j in range(0, input_array.shape[2], self.stride):
                        """
                        注意选定子矩阵时，如果i + self.kernel_size超出了输入尺寸，也会被算进来，所以要提前排除这种情况
                        """
                        if j + self.kernel_size > input_array.shape[2]:
                            break
                        subarray = input_array[b, i:i + self.kernel_size, j:j + self.kernel_size, k]  # 选定子矩阵
                        output[b, i // self.stride, j // self.stride, k] = subarray.max()  # 把最大值存入output相应位置
                        index = np.argmax(subarray)  # 最大值的位置，注意是x*y
                        self.index[b, i + index // self.stride, j + j % self.stride, k] = 1
                        # 除以步长，并取模得到坐标x和y.将坐标标记为1，后面反向传播时用
        return output

    def backward(self, delta):
        # upsample:将delta扩充成输入矩阵的形状，并将index没有记录的位置置为0
        next_delta = np.repeat(np.repeat(delta, self.stride, axis=1), self.stride, axis=2) * self.index
        return next_delta


# 全连接层
class FullyConnectedLayer(object):
    def __init__(self, input_shape, output_num):
        self.input_shape = input_shape
        self.batch_size = self.input_shape[0]
        self.output_shape = (self.batch_size, output_num)

        input_len = reduce(lambda x, y: x * y, input_shape[1:])
        self.weights = np.random.standard_normal((input_len, output_num)) / 100
        self.bias = np.random.standard_normal(output_num) / 100

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, input_array):
        self.flatten_x = input_array.reshape((self.batch_size, -1))  # 将输入flatten
        output = np.dot(self.flatten_x, self.weights) + self.bias
        return output

    def backward(self, delta):
        for i in range(self.batch_size):
            col_x = self.flatten_x[i][:, np.newaxis]  # 把行向量转为(input_shape,1)形状的列向量
            delta_i = delta[i][:, np.newaxis].T  # (1,output_num)
            self.w_grad += np.dot(col_x, delta_i)
            self.b_grad += delta_i.reshape(self.bias.shape)

        next_delta = np.dot(delta, self.weights.T)
        next_delta = np.reshape(next_delta, self.input_shape)
        return next_delta

    def update(self, learning_rate):
        self.weights -= learning_rate * self.w_grad
        self.bias -= learning_rate * self.b_grad

        self.w_grad = np.zeros(self.w_grad.shape)
        self.b_grad = np.zeros(self.b_grad.shape)


def train(X_train, y_train, X_test, y_test):
    output_types = 3  # 分类数
    batch_size = 50
    learning_rate = 1e-4

    train_batch_num = len(y_train) // batch_size  # 训练集batch数量
    test_batch_num = len(y_test) // batch_size
    print("train_batch_num:", train_batch_num)
    print("test_batch_num:", test_batch_num)

    """准备模型"""
    conv1 = ConvolutionLayer((batch_size, 28, 28, 1), output_channels=3, kernel_size=5)
    relu1 = Relu(conv1.output_shape)
    pool1 = PoolingLayer(relu1.output_shape)

    conv2 = ConvolutionLayer(pool1.output_shape, output_channels=3, kernel_size=3)
    relu2 = Relu(conv2.output_shape)
    pool2 = PoolingLayer(relu2.output_shape)

    full = FullyConnectedLayer(pool2.output_shape, output_num=output_types)
    soft = Softmax(full.output_shape)

    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    start = time.time()
    for epoch in range(5):
        """开始训练"""
        epoch_train_loss = 0  # 这一epoch的平均loss
        epoch_train_accuracy = 0  # 这一epoch的平均accuracy
        for batch in tqdm(range(train_batch_num)):
            imgs = X_train[batch * batch_size:(batch + 1) * batch_size].reshape((batch_size, 28, 28, 1))
            labels = y_train[batch * batch_size:(batch + 1) * batch_size]
            # 前向传播
            conv1_out = conv1.forward(imgs)
            relu1_out = relu1.forward(conv1_out)
            pool1_out = pool1.forward(relu1_out)

            conv2_out = conv2.forward(pool1_out)
            relu2_out = relu2.forward(conv2_out)
            pool2_out = pool2.forward(relu2_out)
            full_out = full.forward(pool2_out)
            epoch_train_loss += soft.loss(full_out, labels)

            """训练集acc"""
            cnt = 0
            for j in range(batch_size):
                if labels[j] == np.argmax(full_out[j]):
                    cnt += 1
            accuracy = cnt / batch_size
            epoch_train_accuracy += accuracy
            # if batch % 10 == 0:
            #     print("train batch %d, batch_acc:%t"%(batch, accuracy))

            # 反向传播
            soft.backward()
            conv1.backward(
                relu1.backward(
                    pool1.backward(
                        conv2.backward(
                            relu2.backward(
                                pool2.backward(
                                    full.backward(soft.delta)))))))
            full.update(learning_rate)
            conv2.update(learning_rate)
            conv1.update(learning_rate)

        end = time.time()
        train_loss.append(epoch_train_loss / train_batch_num)
        train_accuracy.append(epoch_train_accuracy / train_batch_num)

        """开始测试"""
        epoch_test_loss = 0
        epoch_test_accuracy = 0
        for batch in range(test_batch_num):
            imgs = X_test[batch * batch_size:(batch + 1) * batch_size].reshape((batch_size, 28, 28, 1))
            labels = y_test[batch * batch_size:(batch + 1) * batch_size]

            conv1_out = conv1.forward(imgs)
            relu1_out = relu1.forward(conv1_out)
            pool1_out = pool1.forward(relu1_out)

            conv2_out = conv2.forward(pool1_out)
            relu2_out = relu2.forward(conv2_out)
            pool2_out = pool2.forward(relu2_out)
            full_out = full.forward(pool2_out)
            epoch_test_loss += soft.loss(full_out, labels)
            """测试集acc"""
            cnt = 0
            for j in range(batch_size):
                if labels[j] == np.argmax(full_out[j]):
                    cnt += 1
            accuracy = cnt / batch_size
            epoch_test_accuracy += accuracy
            print("test batch %d accuracy:%f" % (batch, accuracy))

        test_loss.append(epoch_test_loss / test_batch_num)
        test_accuracy.append(epoch_test_accuracy / test_batch_num)
        print(end - start)

    return train_loss, train_accuracy, test_loss, test_accuracy


if __name__ == "__main__":
    data = np.load("mnist_3_types.npz")
    X_train, y_train, X_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
    train_loss, train_accuracy, test_loss, test_accuracy = train(X_train, y_train, X_test, y_test)
    epoch_num = len(train_loss)
    x = np.arange(epoch_num)
    make_plot(x, train_loss, ylabel="train loss")
    make_plot(x, train_accuracy, ylabel="train accuracy")
    make_plot(x, test_loss, ylabel="test loss")
    make_plot(x, test_accuracy, ylabel="test accuracy")
