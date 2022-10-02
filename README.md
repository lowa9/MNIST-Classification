# MNIST-Classification

Pivot Studio 笔试题第三题

# 使用模型简介:

本模型采用了CNN卷积神经网络来实现识别MNIST手写数字识别,使用了pytorch框架来实现



## 第一部分是数据处理部分

调用的torchvision.datasets里就有MNIST,可以下载文件到本地,一共有四个文件:

* 训练集(train_dataset)
* 训练集标签
* 测试集(test_dataset)
* 测试集标签

具体代码就是datasets.MNISt(root='./data/ mnist',train=True, download=True, transform=transform)

实例化了dataset后就用Dataloader包装起来,载入数据集

transform选择transforms.Normalize()

## 第二部分是建立模型NET

__ init __()里完成以下步骤的封装:

* 首先通过卷积层conv1,该层卷积核为5 * 5,padding为0,步长为1,通过后图像通道数由1变成10,高宽为24 * 24(若n * n为数据原矩阵规模,p为padding大小,f为卷积核大小,s为步长,则输出的矩阵高宽满足下面的式子:)

  ![image](https://user-images.githubusercontent.com/92147115/193443964-6725dd2e-fe5e-49a3-9430-4bfd9e3866ae.png)


* 经过ReLU后通过一层卷积核为2 * 2的Maxpooling,channels不变,高宽变为一半,(batch,10,12,12)

* 经过ReLU后再通过卷积层conv2,该层卷积核为5 * 5,通道数由10变为20,高宽为8 * 8

* 再通过一个卷积核为2×2的最大池化层，通道数不变，高宽变为一半，即维度变成（batch,20,4,4）；

* 最后将输出数据flatten后输入进fully-connected layer中

__ forward __()里完成上述步骤的组装;

## 第三部分是定义loss function以及optimizer

```python
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # lr学习率，momentum冲量
```

参数优化使用了SGD,随机梯度下降,并且使用了momentum动量梯度下降法

## 第四部分是定义训练函数

训练函数利用train_dataset来进行训练模型,函数在该数据集中进行迭代,每一次epoch中都进行forward propagation,back propagation 以及update parameters,经过这些epoch的训练得到模型.具体见代码中注释

## 第五部分是定义测试函数

上一部分得到模型后,需要用test_dataset来进行检测,检查accuracy.在测试集上跑的时候不需要计算梯度,也不用更新参数

```python
with torch.no_grad():#测试集不用算梯度
```

## 最后就训练部分

定义Hyperparameter

```python
batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 10
```

主函数进行10次训练

并且利用matplotlab中的图形化函数画出loss随着训练逐渐降低的图像

