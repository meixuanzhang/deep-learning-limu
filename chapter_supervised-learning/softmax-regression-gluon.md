# 多类逻辑回归 --- 使用Gluon

现在让我们使用gluon来更快速地实现一个多类逻辑回归。

## 获取和读取数据

我们仍然使用FashionMNIST。我们将代码保存在[../utils.py](../utils.py)这样这里不用复制一遍。

```{.python .input  n=6}
import sys
sys.path.append('..')
import utils

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)
```

```{.json .output n=6}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:84: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  label = np.fromstring(fin.read(), dtype=np.uint8).astype(np.int32)\n/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:88: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  data = np.fromstring(fin.read(), dtype=np.uint8)\n"
 }
]
```

```{.python .input  n=5}
train_data
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "<utils.DataLoader at 0x7f917c3bdc50>"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 定义和初始化模型

我们先使用Flatten层将输入数据转成 `batch_size` x `?` 的矩阵，然后输入到10个输出节点的全连接层。照例我们不需要制定每层输入的大小，gluon会做自动推导。

```{.python .input  n=7}
from mxnet import gluon

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(10))#输出
net.initialize()
```

## Softmax和交叉熵损失函数

如果你做了上一章的练习，那么你可能意识到了分开定义Softmax和交叉熵会有数值不稳定性。因此gluon提供一个将这两个函数合起来的数值更稳定的版本

```{.python .input  n=8}
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## 优化

```{.python .input  n=9}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

## 训练

```{.python .input  n=10}
from mxnet import ndarray as nd
from mxnet import autograd

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.798803, Train acc 0.740585, Test acc 0.795673\nEpoch 1. Loss: 0.574877, Train acc 0.810680, Test acc 0.821314\nEpoch 2. Loss: 0.531107, Train acc 0.822967, Test acc 0.832232\nEpoch 3. Loss: 0.506725, Train acc 0.829243, Test acc 0.836839\nEpoch 4. Loss: 0.490466, Train acc 0.833767, Test acc 0.840645\n"
 }
]
```

## 结论

Gluon提供的函数有时候比手工写的数值更稳定。

## 练习

- 再尝试调大下学习率看看？
- 为什么参数都差不多，但gluon版本比从0开始的版本精度更高？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/740)

参数初始化不同
