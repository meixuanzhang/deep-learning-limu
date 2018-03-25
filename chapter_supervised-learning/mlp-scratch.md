# 多层感知机 --- 从0开始

前面我们介绍了包括线性回归和多类逻辑回归的数个模型，它们的一个共同点是全是只含有一个输入层，一个输出层。这一节我们将介绍多层神经网络，就是包含至少一个隐含层的网络。

## 数据获取

我们继续使用FashionMNIST数据集。

```{.python .input  n=2}
import sys
sys.path.append('..')
#..目录的意思，即代表上一级目录。通过这种方式，是的python程序会在上一级找相应的其他python包或者文件。('../..')就是代表上两层的目录
import utils
batch_size = 256 #
train_data, test_data = utils.load_data_fashion_mnist(batch_size)#读取数据，随机，每批读取batch_size，
```

```{.json .output n=2}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:84: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  label = np.fromstring(fin.read(), dtype=np.uint8).astype(np.int32)\n/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:88: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  data = np.fromstring(fin.read(), dtype=np.uint8)\n"
 }
]
```

## 多层感知机

多层感知机与前面介绍的[多类逻辑回归](../chapter_crashcourse/softmax-regression-scratch.md)非常类似，主要的区别是我们在输入层和输出层之间插入了一个到多个隐含层。

![](../img/multilayer-perceptron.png)

这里我们定义一个只有一个隐含层的模型，这个隐含层输出256个节点。

```{.python .input  n=3}
from mxnet import ndarray as nd
from mxnet import random as mxrandom
mxrandom.seed(1)
num_inputs = 28*28 #输入是28×28的图片
num_outputs = 10#输出是10类

num_hidden = 256#中间隐藏层的神经元
weight_scale = .01#标准差
#共三层
W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale)#第一层 
b1 = nd.zeros(num_hidden)

W2 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale)#第二层
b2 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

## 激活函数

如果我们就用线性操作符来构造多层神经网络，那么整个模型仍然只是一个线性函数。这是因为

$$\hat{y} = X \cdot W_1 \cdot W_2 = X \cdot W_3 $$

这里$W_3 = W_1 \cdot W_2$。为了让我们的模型可以拟合非线性函数，我们需要在层之间插入非线性的激活函数。这里我们使用ReLU

$$\textrm{rel}u(x)=\max(x, 0)$$

```{.python .input  n=4}
def relu(X):
    return nd.maximum(X, 0) #两个参数，输入X 跟0比较 第二个参数可以是array
```

## 定义模型

我们的模型就是将层（全连接）和激活函数（Relu）串起来：

```{.python .input  n=5}
def net(X):
    X = X.reshape((-1, num_inputs))#行是样本，列是属性
    h1 = relu(nd.dot(X, W1) + b1)
    output = nd.dot(h1, W2) + b2
    return output
```

## Softmax和交叉熵损失函数

在多类Logistic回归里我们提到分开实现Softmax和交叉熵损失函数可能导致数值不稳定。这里我们直接使用Gluon提供的函数

```{.python .input  n=5}
from mxnet import gluon
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## 训练

训练跟之前一样。

```{.python .input  n=10}
from mxnet import autograd as ag

learning_rate= .5

for epoch in range(5):
    train_loss = 0
    train_acc = 0
    for data, label in train_data:
        with ag.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        utils.SGD(params, learning_rate/batch_size)
            
        train_loss +=nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)
            
    test_acc=utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. LOss: %f, Train acc %f,Test acc %f"%(
        epoch,train_loss/len(train_data),train_acc/len(train_data),test_acc))
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. LOss: 0.799533, Train acc 0.696464,Test acc 0.801382\nEpoch 1. LOss: 0.486227, Train acc 0.820613,Test acc 0.841246\nEpoch 2. LOss: 0.431734, Train acc 0.840946,Test acc 0.858774\nEpoch 3. LOss: 0.400631, Train acc 0.851179,Test acc 0.859876\nEpoch 4. LOss: 0.372336, Train acc 0.862046,Test acc 0.859275\n"
 }
]
```

## 总结

可以看到，加入一个隐含层后我们将精度提升了不少。

## 练习

- 我们使用了 `weight_scale` 来控制权重的初始化值大小，增大或者变小这个值会怎么样？
- 尝试改变 `num_hiddens` 来控制模型的复杂度
- 尝试加入一个新的隐含层

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/739)

```{.python .input  n=9}
mxrandom.seed(1)
num_hidden = 256#中间隐藏层的神经元
weight_scale =1#标准差，部分梯度会比较大，部分比较小，不稳定？
#共三层

W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale)#第一层 
b1 = nd.zeros(num_hidden)

W2 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale)#第二层
b2 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
    
def net(X):
    X = X.reshape((-1, num_inputs))#行是样本，列是属性
    h1 = relu(nd.dot(X, W1) + b1)
    output = nd.dot(h1, W2) + b2
    return output
```

```{.python .input  n=1}
def train(epochs,learning_rate):
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        for data, label in train_data:
            with ag.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            utils.SGD(params, learning_rate/batch_size)
         
            train_loss +=nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
            
        test_acc=utils.evaluate_accuracy(test_data, net)
        print("Epoch %d. LOss: %f, Train acc %f,Test acc %f"%(
            epoch,train_loss/len(train_data),train_acc/len(train_data),test_acc))
```

```{.python .input  n=8}
epochs = 5
learning_rate = 0.5
train(epochs,learning_rate)
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. LOss: 16.202999, Train acc 0.593750,Test acc 0.661258\nEpoch 1. LOss: 0.973461, Train acc 0.674479,Test acc 0.693109\nEpoch 2. LOss: 0.861631, Train acc 0.693343,Test acc 0.703826\nEpoch 3. LOss: 0.796540, Train acc 0.706197,Test acc 0.696815\nEpoch 4. LOss: 0.760128, Train acc 0.711589,Test acc 0.716647\n"
 }
]
```

```{.python .input  n=10}
epochs = 15
learning_rate = 0.5
train(epochs,learning_rate)
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. LOss: 13.635526, Train acc 0.599977,Test acc 0.667969\nEpoch 1. LOss: 0.952625, Train acc 0.678052,Test acc 0.691206\nEpoch 2. LOss: 0.844376, Train acc 0.694511,Test acc 0.669972\nEpoch 3. LOss: 0.781717, Train acc 0.706898,Test acc 0.701923\nEpoch 4. LOss: 0.744957, Train acc 0.714810,Test acc 0.696514\nEpoch 5. LOss: 0.711977, Train acc 0.721254,Test acc 0.713442\nEpoch 6. LOss: 0.683507, Train acc 0.731654,Test acc 0.730068\nEpoch 7. LOss: 0.662703, Train acc 0.738348,Test acc 0.727163\nEpoch 8. LOss: 0.645546, Train acc 0.746378,Test acc 0.744792\nEpoch 9. LOss: 0.623860, Train acc 0.760317,Test acc 0.768429\nEpoch 10. LOss: 0.601382, Train acc 0.773855,Test acc 0.763522\nEpoch 11. LOss: 0.581885, Train acc 0.782235,Test acc 0.774439\nEpoch 12. LOss: 0.574347, Train acc 0.787210,Test acc 0.782652\nEpoch 13. LOss: 0.563454, Train acc 0.791516,Test acc 0.767628\nEpoch 14. LOss: 0.551814, Train acc 0.797660,Test acc 0.794972\n"
 }
]
```

```{.python .input  n=15}
mxrandom.seed(1)
num_hidden = 400#中间隐藏层的神经元
weight_scale =0.1#标准差，部分梯度会比较大，部分比较小，不稳定？
#共三层

W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale)#第一层 
b1 = nd.zeros(num_hidden)

W2 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale)#第二层
b2 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
    
def net(X):
    X = X.reshape((-1, num_inputs))#行是样本，列是属性
    h1 = relu(nd.dot(X, W1) + b1)
    output = nd.dot(h1, W2) + b2
    return output



```

```{.python .input  n=12}
epochs = 5
learning_rate = 0.5
train(epochs,learning_rate)
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. LOss: 0.842848, Train acc 0.750534,Test acc 0.802584\nEpoch 1. LOss: 0.477415, Train acc 0.824369,Test acc 0.811699\nEpoch 2. LOss: 0.409820, Train acc 0.849726,Test acc 0.858874\nEpoch 3. LOss: 0.377604, Train acc 0.862964,Test acc 0.849760\nEpoch 4. LOss: 0.356503, Train acc 0.869057,Test acc 0.842248\n"
 }
]
```

```{.python .input  n=14}
epochs = 10
learning_rate = 0.1
train(epochs,learning_rate)
```

```{.json .output n=14}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. LOss: 0.700879, Train acc 0.765775,Test acc 0.827424\nEpoch 1. LOss: 0.485264, Train acc 0.831798,Test acc 0.844451\nEpoch 2. LOss: 0.441459, Train acc 0.845219,Test acc 0.854367\nEpoch 3. LOss: 0.414097, Train acc 0.854000,Test acc 0.859675\nEpoch 4. LOss: 0.399372, Train acc 0.859008,Test acc 0.860577\nEpoch 5. LOss: 0.383340, Train acc 0.864650,Test acc 0.868890\nEpoch 6. LOss: 0.368800, Train acc 0.869575,Test acc 0.866386\nEpoch 7. LOss: 0.360529, Train acc 0.872780,Test acc 0.865084\nEpoch 8. LOss: 0.350793, Train acc 0.875751,Test acc 0.874499\nEpoch 9. LOss: 0.340321, Train acc 0.878839,Test acc 0.867788\n"
 }
]
```

```{.python .input  n=24}
mxrandom.seed(1)
num_hidden = 400#中间隐藏层的神经元
weight_scale =0.1#标准差，部分梯度会比较大，部分比较小，不稳定？
#共三层

W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale)#第一层 
b1 = nd.zeros(num_hidden)

W2 = nd.random_normal(shape=(num_hidden, num_hidden), scale=weight_scale)#第二层
b2 = nd.zeros(num_hidden)

W3 = nd.random_normal(shape=(num_hidden, num_hidden), scale=weight_scale)#第三层
b3 = nd.zeros(num_hidden)

W4 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale)#第四层
b4 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3, W4, b4]

for param in params:
    param.attach_grad()
    
def net(X):
    X = X.reshape((-1, num_inputs))#行是样本，列是属性
    h1 = relu(nd.dot(X, W1) + b1)
    h2 = relu(nd.dot(h1, W2) + b2)
    h3 = relu(nd.dot(h2, W3) + b3)
    output = nd.dot(h3, W4) + b4
    return output
```

```{.python .input  n=21}
epochs = 10
learning_rate = 0.1
train(epochs,learning_rate)
```

```{.json .output n=21}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. LOss: 0.905621, Train acc 0.765108,Test acc 0.838341\nEpoch 1. LOss: 0.449305, Train acc 0.837874,Test acc 0.847556\nEpoch 2. LOss: 0.394766, Train acc 0.857889,Test acc 0.850461\nEpoch 3. LOss: 0.363227, Train acc 0.867388,Test acc 0.860577\nEpoch 4. LOss: 0.340818, Train acc 0.875601,Test acc 0.872196\nEpoch 5. LOss: 0.324279, Train acc 0.880793,Test acc 0.870793\nEpoch 6. LOss: 0.307765, Train acc 0.887654,Test acc 0.871494\nEpoch 7. LOss: 0.295483, Train acc 0.891760,Test acc 0.861979\nEpoch 8. LOss: 0.281500, Train acc 0.895716,Test acc 0.873498\nEpoch 9. LOss: 0.272927, Train acc 0.898988,Test acc 0.883614\n"
 }
]
```

```{.python .input}
epochs = 30
learning_rate = 0.05
train(epochs,learning_rate)
```

```{.json .output n=None}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. LOss: 0.807574, Train acc 0.770116,Test acc 0.834936\nEpoch 1. LOss: 0.451946, Train acc 0.840695,Test acc 0.824619\nEpoch 2. LOss: 0.404909, Train acc 0.856103,Test acc 0.864283\nEpoch 3. LOss: 0.374155, Train acc 0.866002,Test acc 0.855970\nEpoch 4. LOss: 0.355547, Train acc 0.872579,Test acc 0.868790\nEpoch 5. LOss: 0.338351, Train acc 0.877938,Test acc 0.869191\nEpoch 6. LOss: 0.322708, Train acc 0.884198,Test acc 0.872196\nEpoch 7. LOss: 0.310420, Train acc 0.887103,Test acc 0.872897\nEpoch 8. LOss: 0.298911, Train acc 0.891510,Test acc 0.873197\nEpoch 9. LOss: 0.288441, Train acc 0.896785,Test acc 0.877504\nEpoch 10. LOss: 0.280434, Train acc 0.896918,Test acc 0.875401\nEpoch 11. LOss: 0.271809, Train acc 0.899856,Test acc 0.877905\nEpoch 12. LOss: 0.263349, Train acc 0.904948,Test acc 0.883814\nEpoch 13. LOss: 0.255419, Train acc 0.906901,Test acc 0.884215\nEpoch 14. LOss: 0.250540, Train acc 0.910290,Test acc 0.884315\nEpoch 15. LOss: 0.243096, Train acc 0.912493,Test acc 0.885517\nEpoch 16. LOss: 0.235993, Train acc 0.915064,Test acc 0.883514\nEpoch 17. LOss: 0.230411, Train acc 0.917351,Test acc 0.878606\nEpoch 18. LOss: 0.223276, Train acc 0.919655,Test acc 0.885817\nEpoch 19. LOss: 0.217927, Train acc 0.922676,Test acc 0.889623\nEpoch 20. LOss: 0.213472, Train acc 0.923661,Test acc 0.890024\nEpoch 21. LOss: 0.205115, Train acc 0.926900,Test acc 0.883614\nEpoch 22. LOss: 0.201181, Train acc 0.928602,Test acc 0.891927\nEpoch 23. LOss: 0.198063, Train acc 0.929637,Test acc 0.886418\n"
 }
]
```
