# 设计自定义层

神经网络的一个魅力是它有大量的层，例如全连接、卷积、循环、激活，和各式花样的连接方式。我们之前学到了如何使用Gluon提供的层来构建新的层(`nn.Block`)继而得到神经网络。虽然Gluon提供了大量的[层的定义](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/gluon.html#neural-network-layers)，但我们仍然会遇到现有层不够用的情况。

这时候的一个自然的想法是，我们不是学习了如何只使用基础数值运算包`NDArray`来实现各种的模型吗？它提供了大量的[底层计算函数](https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/ndarray.html)足以实现即使不是100%那也是95%的神经网络吧。

但每次都从头写容易写到怀疑人生。实际上，即使在纯研究的领域里，我们也很少发现纯新的东西，大部分时候是在现有模型的基础上做一些改进。所以很可能大部分是可以沿用前面的而只有一部分是需要自己来实现。

这个教程我们将介绍如何使用底层的`NDArray`接口来实现一个`Gluon`的层，从而可以以后被重复调用。

## 定义一个简单的层

我们先来看如何定义一个简单层，它不需要维护模型参数。事实上这个跟前面介绍的如何使用nn.Block没什么区别。下面代码定义一个层将输入减掉均值。

```{.python .input  n=1}
from mxnet import nd
from mxnet.gluon import nn

class CenteredLayer(nn.Block):
    def __init__(self,**kwargs):
        super(CenteredLayer,self).__init__(**kwargs)
    
    def forward(self,x):
        return x - x.mean()
```

我们可以马上实例化这个层用起来。

```{.python .input  n=2}
layer = CenteredLayer()
layer(nd.array([1,2,3,4,5]))#（）绑定了forward函数
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "\n[-2. -1.  0.  1.  2.]\n<NDArray 5 @cpu(0)>"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们也可以用它来构造更复杂的神经网络：

```{.python .input  n=4}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(128))
    net.add(nn.Dense(10))
    net.add(CenteredLayer())
```

确认下输出的均值确实是0：

```{.python .input  n=5}
net.initialize()
y = net(nd.random.uniform(shape=(4, 8)))
y.mean()
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "\n[6.9849196e-11]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

当然大部分情况你可以看不到一个实实在在的0，而是一个很小的数。例如`5.82076609e-11`。这是因为MXNet默认使用32位float，会带来一定的浮点精度误差。

## 带模型参数的自定义层

虽然`CenteredLayer`可能会告诉实现自定义层大概是什么样子，但它缺少了重要的一块，就是它没有可以学习的模型参数。

记得我们之前访问`Dense`的权重的时候是通过`dense.weight.data()`，这里`weight`是一个`Parameter`的类型。我们可以显示的构建这样的一个参数。

```{.python .input  n=5}
from mxnet import gluon
my_param = gluon.Parameter("exciting_parameter_yay", shape=(3,3))
```

```{.python .input  n=14}
my_param
```

```{.json .output n=14}
[
 {
  "data": {
   "text/plain": "Parameter exciting_parameter_yay (shape=(3, 3), dtype=<class 'numpy.float32'>)"
  },
  "execution_count": 14,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

这里我们创建一个$3\times3$大小的参数并取名为"exciting_parameter_yay"。然后用默认方法初始化打印结果。

```{.python .input  n=15}
my_param.initialize()
(my_param.data(), my_param.grad())
```

```{.json .output n=15}
[
 {
  "data": {
   "text/plain": "(\n [[-0.02548236  0.05326662 -0.01200318]\n  [ 0.05855297 -0.06101935 -0.0396449 ]\n  [ 0.0269461   0.00912645  0.0093242 ]]\n <NDArray 3x3 @cpu(0)>, \n [[0. 0. 0.]\n  [0. 0. 0.]\n  [0. 0. 0.]]\n <NDArray 3x3 @cpu(0)>)"
  },
  "execution_count": 15,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

通常自定义层的时候我们不会直接创建Parameter，而是用过Block自带的一个ParamterDict类型的成员变量`params`，顾名思义，这是一个由字符串名字映射到Parameter的字典。

```{.python .input  n=24}
pd = gluon.ParameterDict(prefix="block1_")
pd.get("exciting_parameter_yay", shape=(3,3))#get 加入一个参数？
pd
```

```{.json .output n=24}
[
 {
  "data": {
   "text/plain": "block1_ (\n  Parameter block1_exciting_parameter_yay (shape=(3, 3), dtype=<class 'numpy.float32'>)\n)"
  },
  "execution_count": 24,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=25}
pd.get("exciting_parameter_yay1", shape=(3,3))
```

```{.json .output n=25}
[
 {
  "data": {
   "text/plain": "Parameter block1_exciting_parameter_yay1 (shape=(3, 3), dtype=<class 'numpy.float32'>)"
  },
  "execution_count": 25,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=26}
pd
```

```{.json .output n=26}
[
 {
  "data": {
   "text/plain": "block1_ (\n  Parameter block1_exciting_parameter_yay (shape=(3, 3), dtype=<class 'numpy.float32'>)\n  Parameter block1_exciting_parameter_yay1 (shape=(3, 3), dtype=<class 'numpy.float32'>)\n)"
  },
  "execution_count": 26,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

现在我们看下如果实现一个跟`Dense`一样功能的层，它概念跟前面的`CenteredLayer`的主要区别是我们在初始函数里通过`params`创建了参数：

```{.python .input  n=17}
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        with self.name_scope():
            self.weight = self.params.get(
                'weight', shape=(in_units, units))
            self.bias = self.params.get('bias', shape=(units,))        

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)
```

我们创建实例化一个对象来看下它的参数，这里我们特意加了前缀`prefix`，这是`nn.Block`初始化函数自带的参数。

```{.python .input  n=27}
dense = MyDense(5, in_units=10, prefix='o_my_dense_')
dense.params
```

```{.json .output n=27}
[
 {
  "data": {
   "text/plain": "o_my_dense_ (\n  Parameter o_my_dense_weight (shape=(10, 5), dtype=<class 'numpy.float32'>)\n  Parameter o_my_dense_bias (shape=(5,), dtype=<class 'numpy.float32'>)\n)"
  },
  "execution_count": 27,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

它的使用跟前面没有什么不一致：

```{.python .input  n=19}
dense.initialize()
dense(nd.random.uniform(shape=(2,10)))
```

```{.json .output n=19}
[
 {
  "data": {
   "text/plain": "\n[[0.         0.         0.03019281 0.09594411 0.13613266]\n [0.         0.         0.00460231 0.10275272 0.15692511]]\n<NDArray 2x5 @cpu(0)>"
  },
  "execution_count": 19,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们构造的层跟Gluon提供的层用起来没太多区别：

```{.python .input  n=28}
net = nn.Sequential()
with net.name_scope():
    net.add(MyDense(32, in_units=64))
    net.add(MyDense(2, in_units=32))
net.initialize()
net(nd.random.uniform(shape=(2,64)))
```

```{.json .output n=28}
[
 {
  "data": {
   "text/plain": "\n[[0.         0.06250843]\n [0.00077505 0.08170694]]\n<NDArray 2x2 @cpu(0)>"
  },
  "execution_count": 28,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=46}
nn.Dense??
```

```{.python .input  n=47}
nn.Block??
```

```{.python .input  n=45}
nn.HybridBlock??
```

```{.python .input  n=30}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(128))
    net.add(nn.Dense(10))
    net.add(CenteredLayer())
```

```{.python .input  n=32}
net[0].collect_params()
```

```{.json .output n=32}
[
 {
  "data": {
   "text/plain": "sequential2_dense0_ (\n  Parameter sequential2_dense0_weight (shape=(128, 0), dtype=<class 'numpy.float32'>)\n  Parameter sequential2_dense0_bias (shape=(128,), dtype=<class 'numpy.float32'>)\n)"
  },
  "execution_count": 32,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=40}
net.initialize()
```

```{.json .output n=40}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/parameter.py:321: UserWarning: Parameter sequential2_dense0_bias is already initialized, ignoring. Set force_reinit=True to re-initialize.\n  \"Set force_reinit=True to re-initialize.\"%self.name)\n/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/parameter.py:321: UserWarning: Parameter sequential2_dense1_bias is already initialized, ignoring. Set force_reinit=True to re-initialize.\n  \"Set force_reinit=True to re-initialize.\"%self.name)\n"
 }
]
```

```{.python .input  n=42}
net[0].weight.data()
```

```{.json .output n=42}
[
 {
  "ename": "DeferredInitializationError",
  "evalue": "Parameter sequential2_dense0_weight has not been initialized yet because initialization was deferred. Actual initialization happens during the first forward pass. Please pass one batch of data through the network before accessing Parameters. You can also avoid deferred initialization by specifying in_units, num_features, etc., for network layers.",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mDeferredInitializationError\u001b[0m               Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-42-59ea5453a5fc>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mnet\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mweight\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
   "\u001b[0;32m~/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/parameter.py\u001b[0m in \u001b[0;36mdata\u001b[0;34m(self, ctx)\u001b[0m\n\u001b[1;32m    388\u001b[0m         \u001b[0mNDArray\u001b[0m \u001b[0mon\u001b[0m \u001b[0mctx\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    389\u001b[0m         \"\"\"\n\u001b[0;32m--> 390\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_check_and_get\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_data\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mctx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    391\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    392\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mlist_data\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/parameter.py\u001b[0m in \u001b[0;36m_check_and_get\u001b[0;34m(self, arr_list, ctx)\u001b[0m\n\u001b[1;32m    182\u001b[0m                 \u001b[0;34m\"Please pass one batch of data through the network before accessing Parameters. \"\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m\\\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    183\u001b[0m                 \u001b[0;34m\"You can also avoid deferred initialization by specifying in_units, \"\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m\\\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 184\u001b[0;31m                 \"num_features, etc., for network layers.\"%(self.name))\n\u001b[0m\u001b[1;32m    185\u001b[0m         raise RuntimeError(\n\u001b[1;32m    186\u001b[0m             \u001b[0;34m\"Parameter %s has not been initialized. Note that \"\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m\\\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mDeferredInitializationError\u001b[0m: Parameter sequential2_dense0_weight has not been initialized yet because initialization was deferred. Actual initialization happens during the first forward pass. Please pass one batch of data through the network before accessing Parameters. You can also avoid deferred initialization by specifying in_units, num_features, etc., for network layers."
  ]
 }
]
```

```{.python .input  n=43}
net(nd.random.uniform(shape=(2,64)))
```

```{.json .output n=43}
[
 {
  "data": {
   "text/plain": "\n[[ 0.10597188  0.11233543 -0.08227646 -0.03601876 -0.08104892  0.03130725\n   0.09440936 -0.05797828 -0.02685231 -0.09661181]\n [ 0.10766582  0.04764751 -0.04038952 -0.02630692 -0.11038025  0.09990734\n   0.1327187  -0.06811512  0.00477565 -0.11076061]]\n<NDArray 2x10 @cpu(0)>"
  },
  "execution_count": 43,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=44}
net[0].weight.data()
```

```{.json .output n=44}
[
 {
  "data": {
   "text/plain": "\n[[ 0.0279271  -0.05373173 -0.02835883 ...  0.06776591  0.05486927\n  -0.03355227]\n [ 0.05647313  0.0380233   0.01031513 ...  0.0654735   0.04788432\n  -0.03103536]\n [-0.02897675 -0.01035685  0.00336936 ... -0.03788597 -0.01165452\n   0.05308829]\n ...\n [ 0.00094164  0.06747638 -0.0619457  ... -0.00642275 -0.06872933\n   0.05425686]\n [-0.0552759   0.06497166 -0.05205983 ...  0.0579171   0.03702006\n   0.02485369]\n [ 0.00333667 -0.06229809 -0.06440519 ...  0.00410174  0.01404112\n   0.05992427]]\n<NDArray 128x64 @cpu(0)>"
  },
  "execution_count": 44,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

仔细的你可能还是注意到了，我们这里指定了输入的大小，而Gluon自带的`Dense`则无需如此。我们已经在前面节介绍过了这个延迟初始化如何使用。但如果实现一个这样的层我们将留到后面介绍了hybridize后。

## 总结

现在我们知道了如何把前面手写过的层全部包装了Gluon能用的Block，之后再用到的时候就可以飞起来了！

## 练习

1. 怎么修改自定义层里参数的默认初始化函数。
1. (这个比较难），在一个代码Cell里面输入`nn.Dense??`，看看它是怎么实现的。为什么它就可以支持延迟初始化了。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/1256)

```{.python .input}
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        with self.name_scope():
            self.weight = self.params.get(
                'weight', shape=(in_units, units))
            self.bias = self.params.get('bias', shape=(units,))
        self.params.initialize(init.One(), force_reinit=True)

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)
```
