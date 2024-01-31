# 根据b站视频教程的训练方式，手写数字识别

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from paddle.vision.datasets import MNIST
from paddle.vision.transforms import ToTensor


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
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.sigmoid(x)
        x = self.max_pool1(x)
        x = F.sigmoid(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.sigmoid(x)
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
def eval_func(model, img):
    model_dict = paddle.load("mnist4.pdparams")
    model.set_state_dict(model_dict)
    model.eval()
    x = paddle.to_tensor(img.reshape([1, 1, 28, 28]).astype("float32"))
    result = F.softmax(model(x)).numpy()[0]
    maxValue = max(result)
    num = 0
    for index, value in enumerate(result):
        if value == maxValue:
            num = index
            break

    print("result: ", result)
    print("正确的数字: ", num)


EPOCH_NUM = 5
model = LeNet(num_classes=10)
# opt = paddle.optimizer.Momentum(learning_rate=0.001, parameters=model.parameters())
# train_loader = paddle.io.DataLoader(MNIST(mode="train", transform=ToTensor()), batch_size=10, shuffle=True)
# valid_loader = paddle.io.DataLoader(MNIST(mode="test", transform=ToTensor()), batch_size=10)
# train(model, opt, train_loader, valid_loader)

# 预测
eval_loader = MNIST(mode="test", transform=ToTensor())
eval_img = np.array(eval_loader[0][0])
# plt.imshow(eval_img.squeeze(), cmap="gray")
# plt.show()
eval_func(model, eval_img)
