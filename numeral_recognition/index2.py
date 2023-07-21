# 定义 SimpleNet 网络结构
import paddle
from PIL import Image
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F
import numpy as np

# 确保从paddle.vision.datasets.MNIST中加载的图像数据是np.ndarray类型
paddle.vision.set_image_backend('cv2')

# MNIST图像高和宽
IMG_ROWS, IMG_COLS = 28, 28


# 多层卷积神经网络实现
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # 定义一层全连接层，输出维度是1
        self.fc = Linear(in_features=980, out_features=10)

    # 加入对每一层输入和输出的尺寸和数据内容的打印，根据check参数决策是否打印每层的参数和输出尺寸
    # 卷积层激活函数使用Relu，全连接层激活函数使用softmax
    def forward(self, inputs, label=None, check_shape=False, check_content=False):
        # 给不同层的输出不同命名，方便调试
        outputs1 = self.conv1(inputs)
        outputs2 = F.relu(outputs1)
        outputs3 = self.max_pool1(outputs2)
        outputs4 = self.conv2(outputs3)
        outputs5 = F.relu(outputs4)
        outputs6 = self.max_pool2(outputs5)
        outputs6 = paddle.reshape(outputs6, [outputs6.shape[0], -1])
        outputs7 = self.fc(outputs6)

        # 选择是否打印神经网络每层的参数尺寸和输出尺寸，验证网络结构是否设置正确
        if check_shape:
            # 打印每层网络设置的超参数-卷积核尺寸，卷积步长，卷积padding，池化核尺寸
            print("\n########## print network layer's superparams ##############")
            print("conv1-- kernel_size:{}, padding:{}, stride:{}".format(self.conv1.weight.shape, self.conv1._padding,
                                                                         self.conv1._stride))
            print("conv2-- kernel_size:{}, padding:{}, stride:{}".format(self.conv2.weight.shape, self.conv2._padding,
                                                                         self.conv2._stride))
            # print("max_pool1-- kernel_size:{}, padding:{}, stride:{}".format(self.max_pool1.pool_size, self.max_pool1.pool_stride, self.max_pool1._stride))
            # print("max_pool2-- kernel_size:{}, padding:{}, stride:{}".format(self.max_pool2.weight.shape, self.max_pool2._padding, self.max_pool2._stride))
            print("fc-- weight_size:{}, bias_size_{}".format(self.fc.weight.shape, self.fc.bias.shape))

            # 打印每层的输出尺寸
            print("\n########## print shape of features of every layer ###############")
            print("inputs_shape: {}".format(inputs.shape))
            print("outputs1_shape: {}".format(outputs1.shape))
            print("outputs2_shape: {}".format(outputs2.shape))
            print("outputs3_shape: {}".format(outputs3.shape))
            print("outputs4_shape: {}".format(outputs4.shape))
            print("outputs5_shape: {}".format(outputs5.shape))
            print("outputs6_shape: {}".format(outputs6.shape))
            print("outputs7_shape: {}".format(outputs7.shape))
            # print("outputs8_shape: {}".format(outputs8.shape))

        # 选择是否打印训练过程中的参数和输出内容，可用于训练过程中的调试
        if check_content:
            # 打印卷积层的参数-卷积核权重，权重参数较多，此处只打印部分参数
            print("\n########## print convolution layer's kernel ###############")
            print("conv1 params -- kernel weights:", self.conv1.weight[0][0])
            print("conv2 params -- kernel weights:", self.conv2.weight[0][0])

            # 创建随机数，随机打印某一个通道的输出值
            idx1 = np.random.randint(0, outputs1.shape[1])
            idx2 = np.random.randint(0, outputs4.shape[1])
            # 打印卷积-池化后的结果，仅打印batch中第一个图像对应的特征
            print("\nThe {}th channel of conv1 layer: ".format(idx1), outputs1[0][idx1])
            print("The {}th channel of conv2 layer: ".format(idx2), outputs4[0][idx2])
            print("The output of last layer:", outputs7[0], '\n')

        # 如果label不是None，则计算分类精度并返回
        if label is not None:
            acc = paddle.metric.accuracy(input=F.softmax(outputs7), label=label)
            return outputs7, acc
        else:
            return outputs7


# 网络结构部分之后的代码，保持不变
def train(model):
    model.train()
    # 调用加载数据的函数，获得MNIST训练数据集
    # 加载训练集 batch_size 设为 16
    train_loader = paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode='train'),
                                        batch_size=16,
                                        shuffle=True)
    # 使用SGD优化器，learning_rate设置为0.01
    opt = paddle.optimizer.Adam(learning_rate=0.01, weight_decay=paddle.regularizer.L2Decay(coeff=1e-5),
                                parameters=model.parameters())
    # 训练5轮
    EPOCH_NUM = 10
    loss_list = []
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据
            imgs, labs = data
            images = []
            labels = []
            for i in range(len(imgs)):
                img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
                label = np.reshape(labs[i], [1]).astype('int64')
                images.append(img)
                labels.append(label)

            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)

            # 前向计算的过程
            # 前向计算的过程，同时拿到模型输出值和分类准确率
            if batch_id == 0 and epoch_id == 0:
                # 打印模型参数和每层输出的尺寸
                predicts, acc = model(images, labels, check_shape=True, check_content=False)
            elif batch_id == 401:
                # 打印模型参数和每层输出的值
                predicts, acc = model(images, labels, check_shape=False, check_content=True)
            else:
                predicts, acc = model(images, labels)

            # 计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)

            # 每训练200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(),
                                                                            acc.numpy()))

            # 后向传播，更新参数的过程
            avg_loss.backward()
            # 最小化loss,更新参数
            opt.step()
            # 清除梯度
            opt.clear_grad()

    # 保存模型参数
    paddle.save(model.state_dict(), 'mnist2.pdparams')
    print("Model has been saved.")


def evaluation(model):
    print('start evaluation .......')
    # 定义预测过程
    params_file_path = 'mnist2.pdparams'
    # 加载模型参数
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)

    model.eval()
    eval_loader = paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode='test'),
                                       batch_size=16,
                                       shuffle=True)

    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(eval_loader()):
        # 准备数据
        imgs, labs = data
        images = []
        labels = []
        for i in range(len(imgs)):
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labs[i], [1]).astype('int64')
            images.append(img)
            labels.append(label)

        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels)

        predicts, acc = model(images, labels)
        loss = F.cross_entropy(input=predicts, label=labels)
        avg_loss = paddle.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))

    # 计算多个batch的平均损失和准确率
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))


# 读取一张本地的样例图片，转变成模型输入的格式
def load_image(img_path):
    # 从img_path中读取图像，并转为灰度图
    im = Image.open(img_path).convert('L')
    im = im.resize((IMG_ROWS, IMG_COLS), Image.LANCZOS)
    im = np.array(im).reshape(1, 1, IMG_ROWS, IMG_COLS).astype(np.float32)
    # 图像归一化
    im = 1.0 - im / 255.
    return im


def validation():
    # 定义预测过程
    params_file_path = 'mnist2.pdparams'
    img_path = './5_1.png'
    # 加载模型参数
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)
    # 灌入数据
    model.eval()
    tensor_img = load_image(img_path)
    # 模型反馈10个分类标签的对应概率
    results = model(paddle.to_tensor(tensor_img))
    # 取概率最大的标签作为预测输出
    lab = np.argsort(results.numpy())
    print("本次预测的数字是: ", lab[0][-1])


model = MNIST()
train(model)
evaluation(model)
validation()
