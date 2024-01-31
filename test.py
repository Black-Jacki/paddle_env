import paddle
import paddle.nn as nn
import paddle.nn.functional as F

source = paddle.rand(shape=[1, 3, 32, 32])

Conv2D = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=2, stride=2)
output = Conv2D(source)
print("Conv2D", output.shape)

MaxPool2D = nn.MaxPool2D(kernel_size=2, stride=2)
output = MaxPool2D(source)
print("MaxPool2D", output.shape)

AvgPool2D = nn.AvgPool2D(kernel_size=2, stride=2)
output = AvgPool2D(source)
print("MaxPool2D", output.shape)

batch_norm = nn.BatchNorm(num_channels=3)
output = batch_norm(source)
print("batch_norm", output.shape)

x = paddle.to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = F.relu(x)
print("x", x)
print("relu", output)
