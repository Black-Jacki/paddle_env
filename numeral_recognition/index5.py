# 用本地图片数据进行训练

import os.path
import paddle
import paddle.nn as nn
import paddle.nn.functional as fn
import numpy as np
import itertools
import matplotlib.pyplot as plt
from paddle.vision.datasets import MNIST
from paddle.vision.transforms import ToTensor
from paddle.io import Dataset, DataLoader
from PIL import Image

EPOCH_NUM = 5
IMG_SIZE = 28
# 模型参数保存路径
PD_PARAMS_FILE_NAME = "mnist5.pdparams"


# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, img) for img in os.listdir(data_dir)]

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 从img_path中读取图像，并转为灰度图
        img = Image.open(img_path).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img).reshape([1, IMG_SIZE, IMG_SIZE]).astype(np.float32)
        # 图像归一化
        img = 1 - img / 255.0
        img = paddle.to_tensor(img)
        label = np.array(os.path.basename(img_path).split(".")[0]).reshape([1]).astype("int64")
        return img, label

    def __len__(self):
        return len(self.image_paths)


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets

        # 计算每个数据集的长度
        self.cumulative_sizes = [0] + [len(dataset) for dataset in self.datasets]
        self.cumulative_sizes = list(itertools.accumulate(self.cumulative_sizes))

    def __getitem__(self, idx):
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                if i == 0:
                    return self.datasets[0][idx]
                else:
                    return self.datasets[i - 1][idx - self.cumulative_sizes[i - 1]]
        raise IndexError('Index out of range')

    def __len__(self):
        return self.cumulative_sizes[-1]


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

            if batch_id % 50 == 0:
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
    paddle.save(model.state_dict(), PD_PARAMS_FILE_NAME)


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


# 创建数据集实例
dataset1 = CustomDataset("./eval_imgs")
dataset2 = CustomDataset("./eval_imgs2")
dataset3 = CustomDataset("./eval_imgs3")
train_dataset = []
eval_dataset = []
for i in range(100):
    train_dataset.append(dataset1)
for i in range(100):
    train_dataset.append(dataset3)
for i in range(20):
    eval_dataset.append(dataset2)

train_loader = DataLoader(ConcatDataset(train_dataset), batch_size=10, shuffle=True)
eval_loader = DataLoader(ConcatDataset(eval_dataset), batch_size=10)

# 查看数据集大小
print('train dataset size:', len(train_dataset))
print('eval dataset size:', len(eval_dataset))

# 遍历数据集示例
# for batch_data in train_loader:
#     images, labels = batch_data
#     print('Batch images shape:', images.shape)
#     print('Batch labels:', labels)
#     break  # 仅遍历一个batch的数据示例

model = LeNet(num_classes=10)
# opt = paddle.optimizer.Adamax(
#     learning_rate=0.001,
#     weight_decay=paddle.regularizer.L2Decay(coeff=1e-5),
#     parameters=model.parameters()
# )
# train(model, opt, train_loader, eval_loader)

# 预测
model_dict = paddle.load(PD_PARAMS_FILE_NAME)
model.set_state_dict(model_dict)
model.eval()

for index in range(0, 10):
    eval_img = np.array(dataset2[index][0])
    label = dataset2[index][1]
    eval_func(model, eval_img, label)
