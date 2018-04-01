# 创建神经网络

前面的教程我们教了大家如何实现线性回归，多类Logistic回归和多层感知机。我们既展示了如何从0开始实现，也提供使用`gluon`的更紧凑的实现。因为前面我们主要关注在模型本身，所以只解释了如何使用`gluon`，但没说明他们是如何工作的。我们使用了`nn.Sequential`，它是`nn.Block`的一个简单形式，但没有深入了解它们。

本教程和接下来几个教程，我们将详细解释如何使用这两个类来定义神经网络、初始化参数、以及保存和读取模型。

我们重新把[多层感知机 --- 使用Gluon](../chapter_supervised-learning/mlp-gluon.md)里的网络定义搬到这里作为开始的例子（为了简单起见，这里我们丢掉了Flatten层）。

```{.python .input  n=1}
from mxnet import nd
from mxnet.gluon import nn

net = nn.Sequential()
with net.name_scope():
    #它的作用是给里面的所有层和参数的名字加上前缀（prefix）使得他们在系统里面独一无二。
    net.add(nn.Dense(256, activation="relu")) 
    net.add(nn.Dense(10))
    #self._children = [nn.Dense..]
    #为了传入forward计算？

print(net)
```

```{.json .output n=1}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential(\n  (0): Dense(None -> 256, Activation(relu))\n  (1): Dense(None -> 10, linear)\n)\n"
 }
]
```

```{.python .input  n=15}
nn.Sequential??
```

```{.python .input  n=55}
nn.Block??
```

```{.python .input  n=56}
nn.Dense??
```

```{.python .input  n=13}
nn.HybridBlock??
```

## 使用 `nn.Block` 来定义

事实上，`nn.Sequential`是`nn.Block`的简单形式。我们先来看下如何使用`nn.Block`来实现同样的网络。

```{.python .input  n=2}
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            #它的作用是给里面的所有层和参数的名字加上前缀（prefix）使得他们在系统里面独一无二。
            self.dense0 = nn.Dense(256)
            self.dense1 = nn.Dense(10)

    def forward(self, x):

        return self.dense1(nd.relu(self.dense0(x)))
    #x是输入数据，self.dense0(x)是输出
```

```{.python .input  n=42}
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP,self).__init__(**kwargs)
        with self.name_scope():
            self.dens0 = nn.Dense(256)
            self.dense1 = nn.Dense(10)
    
    def forward (self,x):
        return self.dense1(nd.relu(self.dense0(x)))    
```

可以看到`nn.Block`的使用是通过创建一个它子类的类，其中至少包含了两个函数。

- `__init__`：创建参数。上面例子我们使用了包含了参数的`dense`层
- `forward()`：定义网络的计算

我们所创建的类的使用跟前面`net`没有太多不一样。

```{.python .input  n=3}
net2 = MLP()
print(net2)
net2.initialize()
x = nd.random.uniform(shape=(4,20))
y = net2(x)
y
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "MLP(\n  (dense0): Dense(None -> 256, linear)\n  (dense1): Dense(None -> 10, linear)\n)\n"
 },
 {
  "data": {
   "text/plain": "\n[[ 0.03126615  0.04562764  0.00039858 -0.08772386 -0.05355632  0.02904574\n   0.08102557 -0.01433946 -0.0422415   0.06047882]\n [ 0.02871901  0.03652266  0.0063005  -0.05650971 -0.07189323  0.08615957\n   0.05951559 -0.06045963 -0.02990259  0.05651   ]\n [ 0.02147349  0.04818897  0.05321142 -0.1261686  -0.06850231  0.09096343\n   0.04064303 -0.05064792 -0.02200241  0.04859561]\n [ 0.03780477  0.0751239   0.03290457 -0.11641112 -0.03254965  0.0586529\n   0.02542158 -0.01697343 -0.00049651  0.05892839]]\n<NDArray 4x10 @cpu(0)>"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input}
nn.Dense
```

如何定义创建和使用`nn.Dense`比较好理解。接下来我们仔细看下`MLP`里面用的其他命令：

- `super(MLP, self).__init__(**kwargs)`：这句话调用`nn.Block`的`__init__`函数，它提供了`prefix`（指定名字）和`params`（指定模型参数）两个参数。我们会之后详细解释如何使用。

- `self.name_scope()`：调用`nn.Block`提供的`name_scope()`函数。`nn.Dense`的定义放在这个`scope`里面。它的作用是给里面的所有层和参数的名字加上前缀（prefix）使得他们在系统里面独一无二。默认自动会自动生成前缀，我们也可以在创建的时候手动指定。推荐在构建网络时，每个层至少在一个`name_scope()`里。

```{.python .input  n=4}
print('default prefix:', net2.dense0.name)

net3 = MLP(prefix='another_mlp_')
print('customized prefix:', net3.dense0.name)
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "default prefix: mlp0_dense0\ncustomized prefix: another_mlp_dense0\n"
 }
]
```

大家会发现这里并没有定义如何求导，或者是`backward()`函数。事实上，系统会使用`autograd`对`forward()`自动生成对应的`backward()`函数。

## `nn.Block`到底是什么东西？

在`gluon`里，`nn.Block`是一个一般化的部件。整个神经网络可以是一个`nn.Block`，单个层也是一个`nn.Block`。我们可以（近似）无限地嵌套`nn.Block`来构建新的`nn.Block`。

`nn.Block`主要提供这个东西

1. 存储参数
2. 描述`forward`如何执行
3. 自动求导

## 那么现在可以解释`nn.Sequential`了吧

`nn.Sequential`是一个`nn.Block`容器，它通过`add`来添加`nn.Block`。它自动生成`forward()`函数，其就是把加进来的`nn.Block`逐一运行。

一个简单的实现是这样的：

```{.python .input}
class Sequential(nn.Block):
    def __init__(self, **kwargs):
        super(Sequential, self).__init__(**kwargs)
    def add(self, block):
        self._children.append(block)
    def forward(self, x):
        for block in self._children:
            x = block(x)
        return x
```

```{.python .input  n=5}
class Sequential(nn.Block):
    def __init__(self,**kwargs):
        super(Sequential,self).__init__(**kwargs)
    def add(self,block):
        self._children.append(block)
    def forward(self,x):
        for block in self._children:
            x=block(x)
        return x
```

可以跟`nn.Sequential`一样的使用这个自定义的类：

```{.python .input}
net4 = Sequential()
with net4.name_scope():
    net4.add(nn.Dense(256, activation="relu"))
    net4.add(nn.Dense(10))

net4.initialize()
y = net4(x)
y
```

```{.python .input  n=8}
net4 = Sequential()
with net.name_scope():
    net4.add(nn.Dense(256,activation="relu"))
    net4.add(nn.Dense(10))

net4.initialize()
y = net4(x)##实际调用是net4.forward()?
y
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "\n[[-0.00411106  0.00781806  0.03506001 -0.01106467  0.09599378 -0.04190594\n   0.01127484 -0.01493318  0.07164907  0.00700367]\n [ 0.01214234  0.02546026  0.03533492 -0.02328116  0.10768863 -0.01672857\n  -0.02653832 -0.03458688  0.0640486  -0.00030122]\n [-0.00452384  0.00228632  0.02761048 -0.05750642  0.10328891 -0.01792852\n  -0.04610601 -0.04085525  0.05824738  0.00033788]\n [-0.00518477 -0.02185423  0.02528594 -0.00436604  0.05142228 -0.02703232\n   0.01939205 -0.03802724  0.02832719 -0.0172073 ]]\n<NDArray 4x10 @cpu(0)>"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

可以看到，`nn.Sequential`的主要好处是定义网络起来更加简单。但`nn.Block`可以提供更加灵活的网络定义。考虑下面这个例子

```{.python .input}
class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(256)
            self.weight = nd.random_uniform(shape=(256,20))

    def forward(self, x):
        x = nd.relu(self.dense(x))
        x = nd.relu(nd.dot(x, self.weight)+1)
        x = nd.relu(self.dense(x))
        return x
```

看到这里我们直接手动创建和初始了权重`weight`，并重复用了`dense`的层。测试一下：

```{.python .input}
fancy_mlp = FancyMLP()
fancy_mlp.initialize()
y = fancy_mlp(x)
print(y.shape)
```

## `nn.Block`和`nn.Sequential`的嵌套使用

现在我们知道了`nn`下面的类基本都是`nn.Block`的子类，他们可以很方便地嵌套使用。

```{.python .input}
class RecMLP(nn.Block):
    def __init__(self, **kwargs):
        super(RecMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        with self.name_scope():
            self.net.add(nn.Dense(256, activation="relu"))
            self.net.add(nn.Dense(128, activation="relu"))
            self.dense = nn.Dense(64)

    def forward(self, x):
        return nd.relu(self.dense(self.net(x)))

rec_mlp = nn.Sequential()
rec_mlp.add(RecMLP())
rec_mlp.add(nn.Dense(10))
print(rec_mlp)
```

## 总结

不知道你同不同意，通过`nn.Block`来定义神经网络跟玩积木很类似。

## 练习

如果把`RecMLP`改成`self.denses = [nn.Dense(256), nn.Dense(128), nn.Dense(64)]`，`forward`就用for loop来实现，会有什么问题吗？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/986)
