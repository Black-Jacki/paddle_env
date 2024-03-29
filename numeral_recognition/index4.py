# 根据b站视频教程的训练方式，手写数字识别

import paddle
import paddle.nn as nn
import paddle.nn.functional as fn
import numpy as np
import matplotlib.pyplot as plt
from paddle.vision.datasets import MNIST
from paddle.vision.transforms import ToTensor
from PIL import Image


# 定义模型
class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5)
        self.max_pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2D(in_channels=16, out_channels=120, kernel_size=4)
        self.fc1 = nn.Linear(in_features=120, out_features=64)
        # self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = fn.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = fn.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = fn.relu(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = fn.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        return x


# 训练方法
def train(model, opt, train_loader, valid_loader):
    print("start training...")
    # paddle.device.set_device("cpu")
    # 训练
    model.train()
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader):
            img = data[0]
            label = data[1]
            # 计算模型输出
            logits = model(img)
            # 计算损失函数
            loss_func = nn.CrossEntropyLoss(reduction="none")
            loss = loss_func(logits, label)
            avg_loss = paddle.mean(loss)

            if batch_id % 500 == 0:
                print("epoch: {}, batch_id: {}, loss is: {:.4f}".format(epoch + 1, batch_id, float(avg_loss.numpy())))

            avg_loss.backward()
            opt.step()
            opt.clear_grad()

        # 验证
        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader):
            img = data[0]
            label = data[1]
            # 计算模型输出
            logits = model(img)
            # 计算损失函数
            loss_func = nn.CrossEntropyLoss(reduction="none")
            loss = loss_func(logits, label)
            acc = paddle.metric.accuracy(logits, label)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())

        print("[validation] accuracy/loss: {:.4f}/{:.4f}".format(np.mean(accuracies), np.mean(losses)))
        model.train()

    # 保存模型参数
    paddle.save(model.state_dict(), "mnist4.pdparams")


# 预测
def eval_func(model, img, label):
    x = paddle.to_tensor(img.reshape([1, 1, 28, 28]).astype("float32"))
    result = fn.softmax(model(x)).numpy()[0]
    maxValue = max(result)
    num = 0
    for index, value in enumerate(result):
        if value == maxValue:
            num = index
            break

    # print("result: ", result)
    print("预测数字: {}, 正确答案: {}, 准确率: {:.2f}%".format(num, label, maxValue * 100))


# 读取一张本地的样例图片，转变成模型输入的格式
def load_image(img_path):
    # 从img_path中读取图像，并转为灰度图
    im = Image.open(img_path).convert("L")
    im = im.resize((IMG_ROWS, IMG_COLS))
    im = np.array(im).reshape([1, 1, IMG_ROWS, IMG_COLS]).astype(np.float32)
    # 图像归一化
    im = 1 - im / 255.0
    return im


EPOCH_NUM = 5
# MNIST图像高和宽
IMG_ROWS, IMG_COLS = 28, 28
model = LeNet(num_classes=10)
# opt = paddle.optimizer.Adamax(
#     learning_rate=0.001,
#     weight_decay=paddle.regularizer.L2Decay(coeff=1e-5),
#     parameters=model.parameters()
# )
# train_loader = paddle.io.DataLoader(MNIST(mode="train", transform=ToTensor()), batch_size=10, shuffle=True)
# valid_loader = paddle.io.DataLoader(MNIST(mode="test", transform=ToTensor()), batch_size=10)
# train(model, opt, train_loader, valid_loader)

# 预测
model_dict = paddle.load("mnist_best.pdparams")
model.set_state_dict(model_dict)
model.eval()

# eval_loader = MNIST(mode="test", transform=ToTensor())
# plt.imshow(eval_img.squeeze(), cmap="gray")
# plt.show()
eval_img_dir = "./eval_imgs2/"

for index in range(0, 10):
    # eval_img = np.array(eval_loader[index][0])
    # label = eval_loader[index][1]
    eval_img = load_image(eval_img_dir + str(index) + ".jpg")
    label = str(index)
    eval_func(model, eval_img, label)
