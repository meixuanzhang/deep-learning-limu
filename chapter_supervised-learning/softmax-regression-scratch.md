# 多类逻辑回归 --- 从0开始

如果你读过了[从0开始的线性回归](linear-regression-scratch.md)，那么最难的部分已经过去了。现在你知道如果读取和操作数据，如何构造目标函数和对它求导，如果定义损失函数，模型和求解。

下面我们来看一个稍微有意思一点的问题，如何使用多类逻辑回归进行多类分类。这个模型跟线性回归的主要区别在于输出节点从一个变成了多个。

![](../img/simple-softmax-net.png)


## 获取数据

演示这个模型的常见数据集是手写数字识别MNIST，它长这个样子。

![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/mnist.png)

这里我们用了一个稍微复杂点的数据集，它跟MNIST非常像，但是内容不再是分类数字，而是服饰。我们通过gluon的data.vision模块自动下载这个数据。

```{.python .input  n=1}
from mxnet import gluon
from mxnet import ndarray as nd

def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')
mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
```

```{.json .output n=1}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:84: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  label = np.fromstring(fin.read(), dtype=np.uint8).astype(np.int32)\n/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:88: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  data = np.fromstring(fin.read(), dtype=np.uint8)\n"
 }
]
```

打印一个样本的形状和它的标号

```{.python .input  n=30}
data, label = mnist_train[0]
('example shape: ', data.shape, 'label:', label)#黑白，第三维channel是1,彩色channel是3
```

我们画出前几个样本的内容，和对应的文本标号

```{.python .input  n=4}
import matplotlib.pyplot as plt

def show_images(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()

def get_text_labels(label):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in label]

data, label = mnist_train[0:9]
show_images(data)
print(get_text_labels(label))
```

## 数据读取

虽然我们可以像前面那样通过`yield`来定义获取批量数据函数，这里我们直接使用gluon.data的DataLoader函数，它每次`yield`一个批量。

```{.python .input  n=2}
batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)# X，y
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)
```

```{.python .input  n=3}
for data, label in train_data:
    print(data, label)
    break
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[[[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  ...\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]]\n\n\n [[[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  ...\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]]\n\n\n [[[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  ...\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]]\n\n\n ...\n\n\n [[[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  ...\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]]\n\n\n [[[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  ...\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]]\n\n\n [[[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.00784314]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  ...\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]\n\n  [[0.        ]\n   [0.        ]\n   [0.        ]\n   ...\n   [0.        ]\n   [0.        ]\n   [0.        ]]]]\n<NDArray 256x28x28x1 @cpu(0)> \n[3. 3. 1. 3. 3. 4. 2. 9. 8. 8. 6. 5. 5. 7. 4. 2. 2. 8. 8. 1. 1. 5. 2. 5.\n 7. 2. 7. 6. 1. 7. 3. 7. 6. 5. 6. 6. 3. 0. 5. 1. 7. 2. 0. 4. 1. 2. 4. 8.\n 1. 9. 0. 8. 4. 7. 2. 6. 4. 5. 0. 7. 7. 6. 7. 7. 0. 4. 0. 3. 2. 6. 0. 4.\n 8. 5. 6. 9. 1. 5. 1. 7. 4. 4. 9. 7. 1. 9. 7. 8. 8. 2. 4. 1. 6. 1. 9. 3.\n 6. 6. 9. 0. 6. 8. 6. 0. 0. 5. 6. 8. 4. 9. 6. 0. 6. 0. 0. 3. 4. 9. 4. 7.\n 9. 0. 4. 1. 6. 9. 1. 6. 7. 2. 8. 2. 4. 5. 7. 6. 6. 9. 3. 8. 3. 7. 0. 4.\n 1. 5. 8. 0. 2. 7. 5. 0. 7. 8. 2. 1. 7. 6. 6. 4. 0. 8. 6. 0. 8. 9. 3. 1.\n 7. 0. 7. 1. 5. 5. 8. 7. 5. 5. 7. 8. 2. 9. 1. 7. 5. 4. 1. 5. 8. 9. 2. 2.\n 3. 3. 2. 1. 5. 1. 3. 1. 3. 5. 4. 1. 5. 9. 4. 9. 9. 2. 9. 2. 2. 8. 4. 1.\n 3. 5. 8. 4. 3. 1. 9. 4. 0. 8. 7. 4. 5. 2. 9. 5. 5. 5. 6. 0. 7. 6. 7. 8.\n 2. 9. 3. 7. 6. 1. 4. 8. 1. 8. 8. 0. 7. 1. 9. 0.]\n<NDArray 256 @cpu(0)>\n"
 }
]
```

注意到这里我们要求每次从训练数据里读取一个由随机样本组成的批量，但测试数据则不需要这个要求。

## 初始化模型参数

跟线性模型一样，每个样本会表示成一个向量。我们这里数据是 28 * 28 大小的图片，所以输入向量的长度是 28 * 28 = 784。因为我们要做多类分类，我们需要对每一个类预测这个样本属于此类的概率。因为这个数据集有10个类型，所以输出应该是长为10的向量。这样，我们需要的权重将是一个 784 * 10 的矩阵：

```{.python .input  n=3}
from mxnet import random as mxrandom
mxrandom.seed(1)
num_inputs = 784#输入维度
num_outputs = 10#输出

W = nd.random_normal(shape=(num_inputs, num_outputs))#矩阵行784，列10维
b = nd.random_normal(shape=num_outputs)

params = [W, b]
```

同之前一样，我们要对模型参数附上梯度：

```{.python .input  n=4}
for param in params:
    param.attach_grad()
```

## 定义模型

在线性回归教程里，我们只需要输出一个标量`yhat`使得尽可能的靠近目标值。但在这里的分类里，我们需要属于每个类别的概率。这些概率需要值为正，而且加起来等于1. 而如果简单的使用 $\boldsymbol{\hat y} = \boldsymbol{W} \boldsymbol{x}$, 我们不能保证这一点。一个通常的做法是通过softmax函数来将任意的输入归一化成合法的概率值。

```{.python .input  n=5}
from mxnet import nd
def softmax(X):
    exp = nd.exp(X)
    # 假设exp是矩阵，这里对行进行求和，并要求保留axis 1，
    # 就是返回 (nrows, 1) 形状的矩阵
    partition = exp.sum(axis=1, keepdims=True)
    return exp / partition
```

可以看到，对于随机输入，我们将每个元素变成了非负数，而且每一行加起来为1。

```{.python .input  n=6}
X = nd.random_normal(shape=(2,5))
X_prob = softmax(X)
print(X_prob)
print(X_prob.sum(axis=1))
```

现在我们可以定义模型了：

```{.python .input  n=7}
def net(X):
    return softmax(nd.dot(X.reshape((-1,num_inputs)), W) + b) 
#返回向量，W是矩阵，列维度是num_inputs,-1系统自动（可改成batch_size ,需要保证X每次样本量是batch_size），将输入变成了行，每行一个样本，W一列是一组权重
```

## 交叉熵损失函数

我们需要定义一个针对预测为概率值的损失函数。其中最常见的是交叉熵损失函数，它将两个概率分布的负交叉熵作为目标值，最小化这个值等价于最大化这两个概率的相似度。

具体来说，我们先将真实标号表示成一个概率分布，例如如果`y=1`，那么其对应的分布就是一个除了第二个元素为1其他全为0的长为10的向量，也就是 `yvec=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]`。那么交叉熵就是`yvec[0]*log(yhat[0])+...+yvec[n]*log(yhat[n])`。注意到`yvec`里面只有一个1，那么前面等价于`log(yhat[y])`。所以我们可以定义这个损失函数了

```{.python .input  n=8}
def cross_entropy(yhat, y):
    return - nd.pick(nd.log(yhat), y)
#一列里取第y个，y是label
```

```{.python .input  n=5}
x = [[ 1.,  2.],
     [ 3.,  4.],
     [ 5.,  6.]]
x=nd.array(x)
index=nd.array([0,1])
nd.pick(x, index,axis=0)
```

## 计算精度

给定一个概率输出，我们将预测概率最高的那个类作为预测的类，然后通过比较真实标号我们可以计算精度：

```{.python .input  n=9}
def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()
```

我们可以评估一个模型在这个数据上的精度。（这两个函数我们之后也会用到，所以也都保存在[../utils.py](../utils.py)。）

```{.python .input  n=10}
def evaluate_accuracy(data_iterator, net):
    acc = 0.
    for data, label in data_iterator:
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iterator)
```

因为我们随机初始化了模型，所以这个模型的精度应该大概是`1/num_outputs = 0.1`.

```{.python .input  n=14}
evaluate_accuracy(test_data, net)
```

## 训练

训练代码跟前面的线性回归非常相似：

```{.python .input  n=29}
import sys
sys.path.append('..')
from utils import SGD
from mxnet import autograd

learning_rate = .1

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label)
        loss.backward()
        # 将梯度做平均，这样学习率会对batch size不那么敏感
        SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

```{.python .input  n=11}
import sys
sys.path.append('..')
from utils import SGD
from mxnet import autograd

learning_rate = .5

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label)
        loss.backward()
        # 将梯度做平均，这样学习率会对batch size不那么敏感
        SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

## 预测

训练完成后，现在我们可以演示对输入图片的标号的预测

```{.python .input  n=16}
data, label = mnist_test[0:9]
show_images(data)
print('true labels')
print(get_text_labels(label))

predicted_labels = net(data).argmax(axis=1)
print('predicted labels')
print(get_text_labels(predicted_labels.asnumpy()))
```

## 结论

与前面的线性回归相比，你会发现多类逻辑回归教程的结构跟其非常相似：获取数据、定义模型及优化算法和求解。事实上，几乎所有的实际神经网络应用都有着同样结构。他们的主要区别在于模型的类型和数据的规模。每一两年会有一个新的优化算法出来，但它们基本都是随机梯度下降的变种。

## 练习

尝试增大学习率，你会马上发现结果变得很糟糕，精度基本徘徊在随机的0.1左右。这是为什么呢？提示：

- 打印下output看看是不是有什么异常
- 前面线性回归还好好的，这里我们在net()里加了什么呢？
- 如果给exp输入个很大的数会怎么样？
- 即使解决exp的问题，求出来的导数是不是还是不稳定？

请仔细想想再去对比下我们小伙伴之一@[pluskid](https://github.com/pluskid)早年写的一篇[blog解释这个问题](http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/)，看看你想的是不是不一样。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/741)



指数函数靠近0梯度会趋向无限大？
数值误差是上溢(overflow)。当大数值被近似成或-overflow就发生了。计算机中通常会把这些无限数表示成一个非数值(not-a-number values)。
当c是一个非常大的负数时，exp(c)就会underflow。这意味着softmax的分母为0，所有最终的值是不确定的。当c是一个很大的正数时，exp(c)会overflow，结果同样会变得不确定。
梯度下降在接近底部会有来回震荡?
exp输入很大会导致超出计算范围？
