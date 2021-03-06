# 创建神经网络

前面的教程我们教了大家如何实现线性回归，多类Logistic回归和多层感知机。我们既展示了如何从0开始实现，也提供使用`gluon`的更紧凑的实现。因为前面我们主要关注在模型本身，所以只解释了如何使用`gluon`，但没说明他们是如何工作的。我们使用了`nn.Sequential`，它是`nn.Block`的一个简单形式，但没有深入了解它们。

本教程和接下来几个教程，我们将详细解释如何使用这两个类来定义神经网络、初始化参数、以及保存和读取模型。

我们重新把[多层感知机 --- 使用Gluon](../chapter_supervised-learning/mlp-gluon.md)里的网络定义搬到这里作为开始的例子（为了简单起见，这里我们丢掉了Flatten层）。

```{.python .input  n=76}
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

```{.json .output n=76}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential(\n  (0): Dense(None -> 256, Activation(relu))\n  (1): Dense(None -> 10, linear)\n)\n"
 }
]
```

```{.python .input  n=87}
nn.Sequential??
```

```{.python .input  n=85}
nn.Block??
```

```{.python .input  n=86}
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

```{.python .input  n=101}
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

可以跟`nn.Sequential`一样的使用这个自定义的类：

```{.python .input  n=102}
net4 = Sequential()
with net.name_scope():
    net4.add(nn.Dense(256,activation="relu"))
    net4.add(nn.Dense(10))

net4.initialize()
y = net4(x)##实际调用是net4.forward()?
y
```

```{.json .output n=102}
[
 {
  "data": {
   "text/plain": "\n[[ 0.00841137 -0.0489683  -0.01615157 -0.00888151  0.00140648  0.03107938\n   0.0066022   0.03934433  0.03551184 -0.04462051]\n [ 0.0310174  -0.01866329 -0.03238708 -0.02323047  0.01515523  0.08438477\n  -0.01215139  0.03013801  0.06884266 -0.05892515]\n [ 0.01259469 -0.00025952  0.021398   -0.04749968  0.00458046  0.06876937\n   0.04171674  0.04148455  0.05035593 -0.01429048]\n [ 0.01869043  0.0338157   0.00929088 -0.01238955 -0.02665132  0.03643298\n   0.01158281  0.03388916 -0.00470036 -0.06287378]]\n<NDArray 4x10 @cpu(0)>"
  },
  "execution_count": 102,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=103}
print(net4)
```

```{.json .output n=103}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential(\n\n)\n"
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

```{.python .input  n=9}
class FancyMLP(nn.Block):
    def __init__(self,**kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(256)
            self.weight = nd.random_uniform(shape=(256,20))
            
    def forward(self,x):
        x=nd.relu(self.dense(x))
        x=nd.relu(nd.dot(x,self.weight)+1)
        x=nd.relu(self.dense(x))
        return x
```

看到这里我们直接手动创建和初始了权重`weight`，并重复用了`dense`的层。测试一下：

```{.python .input  n=10}
fancy_mlp = FancyMLP()
fancy_mlp.initialize()
y = fancy_mlp(x)
print(y.shape)
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(4, 256)\n"
 }
]
```

## `nn.Block`和`nn.Sequential`的嵌套使用

现在我们知道了`nn`下面的类基本都是`nn.Block`的子类，他们可以很方便地嵌套使用。

```{.python .input  n=11}
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

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential(\n  (0): RecMLP(\n    (net): Sequential(\n      (0): Dense(None -> 256, Activation(relu))\n      (1): Dense(None -> 128, Activation(relu))\n    )\n    (dense): Dense(None -> 64, linear)\n  )\n  (1): Dense(None -> 10, linear)\n)\n"
 }
]
```

```{.python .input  n=78}
class RecMLP(nn.Block):
    def __init__(self,**kwargs):
        super(RecMLP,self).__init__(**kwargs)
        with self.name_scope():
            self.dense=[nn.Dense(256), nn.Dense(128), nn.Dense(64)]
            self.dense0= nn.Dense(64)
    def forward(self,x):
        for dense in self.denses:
            x=dense(x)
        return x
```

```{.python .input  n=79}
net = RecMLP()
print(net)
net.initialize()
print(net)
```

```{.json .output n=79}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "RecMLP(\n  (dense0): Dense(None -> 64, linear)\n)\nRecMLP(\n  (dense0): Dense(None -> 64, linear)\n)\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/block.py:229: UserWarning: \"RecMLP.dense\" is a container with Blocks. Note that Blocks inside the list, tuple or dict will not be registered automatically. Make sure to register them using register_child() or switching to nn.Sequential/nn.HybridSequential instead. \n  .format(name=self.__class__.__name__ + \".\" + k))\n"
 }
]
```

```{.python .input  n=98}
class RecMLP(nn.Block):
    def __init__(self,**kwargs):
        super(RecMLP,self).__init__(**kwargs)
        self.dense0= nn.Dense(64)
        self._children.append(nn.Dense(65))

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        modstr = '\n'.join(['  ({key}): {block}'.format(key=key,
                                                        block=block.__repr__())
                            for key, block in enumerate(self._children)
                            if isinstance(block, nn.Block)])
        return s.format(name=self.__class__.__name__,
                        modstr=modstr)
```

```{.python .input  n=99}
net = RecMLP()
```

```{.python .input  n=100}
print(net)
```

```{.json .output n=100}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "RecMLP(\n  (0): Dense(None -> 64, linear)\n  (1): Dense(None -> 65, linear)\n)\n"
 }
]
```

## 总结

不知道你同不同意，通过`nn.Block`来定义神经网络跟玩积木很类似。

## 练习

如果把`RecMLP`改成`self.denses = [nn.Dense(256), nn.Dense(128), nn.Dense(64)]`，`forward`就用for loop来实现，会有什么问题吗？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/986)

```{.python .input}
看源码后发现原因为: [nn.Dense(256), nn.Dense(128), nn.Dense(64)] 的 type 是 list, 而不是 Block, 这样就不会被自动注册到 Block 类的 self._children 属性, 导致 initialize 时在 self._children 找不到神经元, 无法初始化参数.

    当执行 self.xxx = yyy 时, __setattr__ 方法会检测 yyy 是否为 Block 类型, 如果是则添加到 self._children 列表中.
    当执行 initialize() 时, 会从 self._children 中找神经元.

详情见源码 Block 类的 __setattr__ 和 initialize 方法:
    
https://www.cnblogs.com/elie/p/6685429.html
```
