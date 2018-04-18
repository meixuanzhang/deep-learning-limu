# 初始化模型参数

我们仍然用MLP这个例子来详细解释如何初始化模型参数。

```{.python .input  n=1}
from mxnet.gluon import nn
from mxnet import nd

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(4, activation="relu"))
        net.add(nn.Dense(2))
    return net

x = nd.random.uniform(shape=(3,5))
```

我们知道如果不`initialize()`直接跑forward，那么系统会抱怨说参数没有初始化。

```{.python .input  n=3}
import sys
try:
    net = get_net()
    net(x)
except RuntimeError as err:
    sys.stderr.write(str(err))
```

```{.json .output n=3}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Parameter sequential0_dense0_weight has not been initialized. Note that you should initialize parameters and create Trainer with Block.collect_params() instead of Block.params because the later does not include Parameters of nested child Blocks"
 }
]
```

正确的打开方式是这样

```{.python .input  n=4}
net.initialize()
net(x)
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "\n[[0.00212593 0.00365805]\n [0.00161272 0.00441845]\n [0.00204872 0.00352518]]\n<NDArray 3x2 @cpu(0)>"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 访问模型参数

之前我们提到过可以通过`weight`和`bias`访问`Dense`的参数，他们是`Parameter`这个类：

```{.python .input  n=5}
w = net[0].weight
b = net[0].bias
print('name: ', net[0].name, '\nweight: ', w, '\nbias: ', b)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "name:  sequential0_dense0 \nweight:  Parameter sequential0_dense0_weight (shape=(4, 5), dtype=<class 'numpy.float32'>) \nbias:  Parameter sequential0_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n"
 }
]
```

然后我们可以通过`data`来访问参数，`grad`来访问对应的梯度

```{.python .input  n=6}
print('weight:', w.data())
print('weight gradient', w.grad())
print('bias:', b.data())
print('bias gradient', b.grad())
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "weight: \n[[-0.06206018  0.06491279 -0.03182812 -0.01631819 -0.00312688]\n [ 0.0408415   0.04370362  0.00404529 -0.0028032   0.00952624]\n [-0.01501013  0.05958354  0.04705103 -0.06005495 -0.02276454]\n [-0.0578019   0.02074406 -0.06716943 -0.01844618  0.04656678]]\n<NDArray 4x5 @cpu(0)>\nweight gradient \n[[0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]]\n<NDArray 4x5 @cpu(0)>\nbias: \n[0. 0. 0. 0.]\n<NDArray 4 @cpu(0)>\nbias gradient \n[0. 0. 0. 0.]\n<NDArray 4 @cpu(0)>\n"
 }
]
```

我们也可以通过`collect_params`来访问Block里面所有的参数（这个会包括所有的子Block）。它会返回一个名字到对应Parameter的dict。既可以用正常`[]`来访问参数，也可以用`get()`，它不需要填写名字的前缀。

```{.python .input  n=7}
params = net.collect_params() #返回的是class mxnet.gluon.ParameterDict 
print(params)
print(params['sequential0_dense0_bias'].data())
print(params.get('dense0_weight').data())
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "sequential0_ (\n  Parameter sequential0_dense0_weight (shape=(4, 5), dtype=<class 'numpy.float32'>)\n  Parameter sequential0_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n  Parameter sequential0_dense1_weight (shape=(2, 4), dtype=<class 'numpy.float32'>)\n  Parameter sequential0_dense1_bias (shape=(2,), dtype=<class 'numpy.float32'>)\n)\n\n[0. 0. 0. 0.]\n<NDArray 4 @cpu(0)>\n\n[[-0.06206018  0.06491279 -0.03182812 -0.01631819 -0.00312688]\n [ 0.0408415   0.04370362  0.00404529 -0.0028032   0.00952624]\n [-0.01501013  0.05958354  0.04705103 -0.06005495 -0.02276454]\n [-0.0578019   0.02074406 -0.06716943 -0.01844618  0.04656678]]\n<NDArray 4x5 @cpu(0)>\n"
 }
]
```

```{.python .input  n=8}
params = net.collect_params()  #返回的是class mxnet.gluon.ParameterDict 
print(params)
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "sequential0_ (\n  Parameter sequential0_dense0_weight (shape=(4, 5), dtype=<class 'numpy.float32'>)\n  Parameter sequential0_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n  Parameter sequential0_dense1_weight (shape=(2, 4), dtype=<class 'numpy.float32'>)\n  Parameter sequential0_dense1_bias (shape=(2,), dtype=<class 'numpy.float32'>)\n)\n"
 }
]
```

## 使用不同的初始函数来初始化

我们一直在使用默认的`initialize`来初始化权重（除了指定GPU `ctx`外）。它会把所有权重初始化成在`[-0.07, 0.07]`之间均匀分布的随机数。我们可以使用别的初始化方法。例如使用均值为0，方差为0.02的正态分布

```{.python .input  n=15}
from mxnet import init
params.initialize(init=init.Normal(sigma=0.02), force_reinit=True)#Pclass mxnet.gluon.Parameter   (class mxnet.initializer.Initializer)init
print(net[0].weight.data(), net[0].bias.data())
```

```{.json .output n=15}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[-0.01884949  0.02661467 -0.00985822 -0.01562379  0.02966928]\n [-0.00744103  0.00702562  0.00869039 -0.03277136  0.02082133]\n [-0.03126733  0.00902533  0.03002612  0.00334209 -0.02473557]\n [-0.01115702  0.02362529 -0.00702631  0.01340065 -0.03474445]]\n<NDArray 4x5 @cpu(0)> \n[0. 0. 0. 0.]\n<NDArray 4 @cpu(0)>\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/parameter.py:321: UserWarning: Parameter sequential0_dense0_weight is already initialized, ignoring. Set force_reinit=True to re-initialize.\n  \"Set force_reinit=True to re-initialize.\"%self.name)\n/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/parameter.py:321: UserWarning: Parameter sequential0_dense0_bias is already initialized, ignoring. Set force_reinit=True to re-initialize.\n  \"Set force_reinit=True to re-initialize.\"%self.name)\n/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/parameter.py:321: UserWarning: Parameter sequential0_dense1_weight is already initialized, ignoring. Set force_reinit=True to re-initialize.\n  \"Set force_reinit=True to re-initialize.\"%self.name)\n/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/parameter.py:321: UserWarning: Parameter sequential0_dense1_bias is already initialized, ignoring. Set force_reinit=True to re-initialize.\n  \"Set force_reinit=True to re-initialize.\"%self.name)\n"
 }
]
```

看得更加清楚点：

```{.python .input  n=16}
params.initialize(init=init.One(), force_reinit=True)
print(net[0].weight.data(), net[0].bias.data())
```

```{.json .output n=16}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[1. 1. 1. 1. 1.]\n [1. 1. 1. 1. 1.]\n [1. 1. 1. 1. 1.]\n [1. 1. 1. 1. 1.]]\n<NDArray 4x5 @cpu(0)> \n[0. 0. 0. 0.]\n<NDArray 4 @cpu(0)>\n"
 }
]
```

```{.python .input}
#net.initialize调用
def initialize(self, init=initializer.Uniform(), ctx=None, verbose=False):
    self.collect_params().initialize(init, ctx, verbose)
```

更多的方法参见[init的API](https://mxnet.incubator.apache.org/api/python/optimization.html#the-mxnet-initializer-package). 

## 延后的初始化

我们之前提到过Gluon的一个便利的地方是模型定义的时候不需要指定输入的大小，在之后做forward的时候会自动推测参数的大小。我们具体来看这是怎么工作的。

新创建一个网络，然后打印参数。你会发现两个全连接层的权重的形状里都有0。 这是因为在不知道输入数据的情况下，我们无法判断它们的形状。

```{.python .input  n=12}
net = get_net()
net.collect_params()
```

```{.json .output n=12}
[
 {
  "data": {
   "text/plain": "sequential1_ (\n  Parameter sequential1_dense0_weight (shape=(4, 0), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense1_weight (shape=(2, 0), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense1_bias (shape=(2,), dtype=<class 'numpy.float32'>)\n)"
  },
  "execution_count": 12,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

然后我们初始化

```{.python .input  n=13}
net.initialize()
net.collect_params()
```

```{.json .output n=13}
[
 {
  "data": {
   "text/plain": "sequential1_ (\n  Parameter sequential1_dense0_weight (shape=(4, 0), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense1_weight (shape=(2, 0), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense1_bias (shape=(2,), dtype=<class 'numpy.float32'>)\n)"
  },
  "execution_count": 13,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

你会看到我们形状并没有发生变化，这是因为我们仍然不能确定权重形状。真正的初始化发生在我们看到数据时。

```{.python .input  n=16}
net(x)
net.collect_params()
```

```{.json .output n=16}
[
 {
  "data": {
   "text/plain": "sequential1_ (\n  Parameter sequential1_dense0_weight (shape=(4, 5), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense1_weight (shape=(2, 4), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense1_bias (shape=(2,), dtype=<class 'numpy.float32'>)\n)"
  },
  "execution_count": 16,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

这时候我们看到shape里面的0被填上正确的值了。

## 共享模型参数

有时候我们想在层之间共享同一份参数，我们可以通过Block的`params`输出参数来手动指定参数，而不是让系统自动生成。

```{.python .input  n=49}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(4, activation="relu"))
    net.add(nn.Dense(4, activation="relu"))
    net.add(nn.Dense(4, activation="relu", params=net[-1].params))#当前net[-1]是上一层
    net.add(nn.Dense(2))
```

初始化然后打印

```{.python .input  n=54}
net.initialize() #默认weight初始化是均匀分布继承了block的initialize？？
net(x)
print(net[0].weight.data())
print(net[1].weight.data())
print(net[2].weight.data())
print(net[3].weight.data())
```

```{.json .output n=54}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.02479142  0.00512109  0.01498631  0.05553398  0.03005757]\n [ 0.06864745 -0.0042704  -0.03963442 -0.00615795  0.02283095]\n [ 0.05689853 -0.03313487 -0.05078914 -0.06710886 -0.0379093 ]\n [ 0.03617301  0.05342196 -0.0251976   0.0566195  -0.01631505]]\n<NDArray 4x5 @cpu(0)>\n\n[[ 0.02040984  0.01236439 -0.02454438  0.04634678]\n [ 0.00275957  0.01805746 -0.06999225  0.0521711 ]\n [-0.02633957 -0.03170411 -0.01043678  0.04172656]\n [ 0.05394727 -0.04401097  0.02518312  0.06339083]]\n<NDArray 4x4 @cpu(0)>\n\n[[ 0.02040984  0.01236439 -0.02454438  0.04634678]\n [ 0.00275957  0.01805746 -0.06999225  0.0521711 ]\n [-0.02633957 -0.03170411 -0.01043678  0.04172656]\n [ 0.05394727 -0.04401097  0.02518312  0.06339083]]\n<NDArray 4x4 @cpu(0)>\n\n[[-0.00614183  0.02624836 -0.00232279 -0.03982893]\n [ 0.04042352  0.06263188 -0.03787814  0.03231981]]\n<NDArray 2x4 @cpu(0)>\n"
 }
]
```

## 自定义初始化方法

下面我们自定义一个初始化方法。它通过重载`_init_weight`来实现不同的初始化方法。（注意到Gluon里面`bias`都是默认初始化成0）

```{.python .input  n=77}
class MyInit(init.Initializer):#class mxnet.initializer.Initializer??重载 init=..也会重载??
    def __init__(self):
        super(MyInit, self).__init__()
        self._verbose = True
    def _init_weight(self, _, arr):
        # 初始化权重，使用out=arr后我们不需指定形状
        #print('init weight', arr.shape)
        nd.random.uniform(low=5, high=10, out=arr)
    def _init_bias(self, _, arr):
        #print('init_bias',arr.shape)
        arr[:] = 1.0
        #print('init_bias',arr.shape)
```

```{.python .input}
https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/initializer.py
```

```{.python .input  n=47}
net = get_net()
net.initialize(MyInit())
net(x)
net[0].bias.data()
```

```{.json .output n=47}
[
 {
  "data": {
   "text/plain": "\n[0. 0. 0. 0.]\n<NDArray 4 @cpu(0)>"
  },
  "execution_count": 47,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=48}
net[0].weight.data()
```

```{.json .output n=48}
[
 {
  "data": {
   "text/plain": "\n[[7.8478675 8.697754  7.2605453 7.4522943 9.851183 ]\n [6.137073  8.402723  6.2717824 5.426478  5.290146 ]\n [5.2820916 7.172083  7.4391885 6.5589795 9.405023 ]\n [8.481717  9.882022  6.888759  8.088289  5.8980184]]\n<NDArray 4x5 @cpu(0)>"
  },
  "execution_count": 48,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=45}
net[0].bias.data()
```

```{.json .output n=45}
[
 {
  "data": {
   "text/plain": "\n[0. 0. 0. 0.]\n<NDArray 4 @cpu(0)>"
  },
  "execution_count": 45,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

当然我们也可以通过`Parameter.set_data`来直接改写权重。注意到由于有延后初始化，所以我们通常可以通过调用一次`net(x)`来确定权重的形状先。

```{.python .input  n=17}
net = get_net()
net.initialize()
net(x)

print('default weight:', net[1].weight.data())

w = net[1].weight
w.set_data(nd.ones(w.shape))

print('init to all 1s:', net[1].weight.data())
```

```{.json .output n=17}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "default weight: \n[[-0.02043347  0.01272219  0.00725428  0.01040554]\n [-0.06529249  0.02144811  0.06565464  0.02129445]]\n<NDArray 2x4 @cpu(0)>\ninit to all 1s: \n[[1. 1. 1. 1.]\n [1. 1. 1. 1.]]\n<NDArray 2x4 @cpu(0)>\n"
 }
]
```

## 总结

我们可以很灵活地访问和修改模型参数。

## 练习

1. 研究下`net.collect_params()`返回的是什么？`net.params`呢？
1. 如何对每个层使用不同的初始化函数
1. 如果两个层共用一个参数，那么求梯度的时候会发生什么？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/987)

```{.python .input}
net.colleact_params()  返回ParameterDict
```

```{.python .input  n=2}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(4, activation="relu"))
    net.add(nn.Dense(2))
```

```{.python .input  n=6}
from mxnet import init
net[0].params.initialize(init=init.Uniform())
net[1].params.initialize(init=init.Constant(3))
net(x)
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "\n[[0.16377717 0.16377717]\n [0.2605431  0.2605431 ]\n [0.15782857 0.15782857]]\n<NDArray 3x2 @cpu(0)>"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=10}
x
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "\n[[0.5488135  0.5928446  0.71518934 0.84426576 0.60276335]\n [0.8579456  0.5448832  0.8472517  0.4236548  0.6235637 ]\n [0.6458941  0.3843817  0.4375872  0.2975346  0.891773  ]]\n<NDArray 3x5 @cpu(0)>"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=9}
print(net.collect_params())
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "sequential0_ (\n  Parameter sequential0_dense0_weight (shape=(4, 5), dtype=<class 'numpy.float32'>)\n  Parameter sequential0_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n  Parameter sequential0_dense1_weight (shape=(2, 4), dtype=<class 'numpy.float32'>)\n  Parameter sequential0_dense1_bias (shape=(2,), dtype=<class 'numpy.float32'>)\n)\n"
 }
]
```

```{.python .input  n=7}
net[1].weight.data()
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "\n[[3. 3. 3. 3.]\n [3. 3. 3. 3.]]\n<NDArray 2x4 @cpu(0)>"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=8}
net[0].initialize(init=init.Uniform(), verbose=True)
net[1].initialize(init=init.Constant(3))
net(x)
```

```{.json .output n=8}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/parameter.py:321: UserWarning: Parameter sequential0_dense0_weight is already initialized, ignoring. Set force_reinit=True to re-initialize.\n  \"Set force_reinit=True to re-initialize.\"%self.name)\n/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/parameter.py:321: UserWarning: Parameter sequential0_dense0_bias is already initialized, ignoring. Set force_reinit=True to re-initialize.\n  \"Set force_reinit=True to re-initialize.\"%self.name)\n/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/parameter.py:321: UserWarning: Parameter sequential0_dense1_weight is already initialized, ignoring. Set force_reinit=True to re-initialize.\n  \"Set force_reinit=True to re-initialize.\"%self.name)\n/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/parameter.py:321: UserWarning: Parameter sequential0_dense1_bias is already initialized, ignoring. Set force_reinit=True to re-initialize.\n  \"Set force_reinit=True to re-initialize.\"%self.name)\n"
 },
 {
  "data": {
   "text/plain": "\n[[0.16377717 0.16377717]\n [0.2605431  0.2605431 ]\n [0.15782857 0.15782857]]\n<NDArray 3x2 @cpu(0)>"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=31}
print(net)
```

```{.json .output n=31}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential(\n  (0): Dense(5 -> 4, Activation(relu))\n  (1): Dense(4 -> 2, linear)\n)\n"
 }
]
```

```{.python .input  n=66}
net[0].weight.data()
```

```{.json .output n=66}
[
 {
  "data": {
   "text/plain": "\n[[ 0.01498631  0.05553398  0.03005757  0.06864745 -0.0042704 ]\n [-0.03963442 -0.00615795  0.02283095  0.05689853 -0.03313487]\n [-0.05078914 -0.06710886 -0.0379093   0.03617301  0.05342196]\n [-0.0251976   0.0566195  -0.01631505  0.02040984  0.01236439]]\n<NDArray 4x5 @cpu(0)>"
  },
  "execution_count": 66,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=67}
net[1].weight.data()
```

```{.json .output n=67}
[
 {
  "data": {
   "text/plain": "\n[[3. 3. 3. 3.]\n [3. 3. 3. 3.]]\n<NDArray 2x4 @cpu(0)>"
  },
  "execution_count": 67,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=75}
net[0].params
```

```{.json .output n=75}
[
 {
  "data": {
   "text/plain": "mxnet.gluon.parameter.ParameterDict"
  },
  "execution_count": 75,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=80}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(4, in_units=4, activation="relu"))
    net.add(nn.Dense(4, in_units=4, activation="relu", params=net[-1].params))
    net.add(nn.Dense(1, in_units=4))
    
net.initialize(MyInit())
print(net[0].weight.data())
print(net[1].weight.data())
print(net[2].weight.data())
```

```{.json .output n=80}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[8.522371  6.9802985 7.315751  7.8271065]\n [9.202143  5.916399  6.0243287 5.724239 ]\n [5.8247943 7.4402814 5.6241655 6.778064 ]\n [8.610403  9.70216   5.1522646 8.826626 ]]\n<NDArray 4x4 @cpu(0)>\n\n[[8.522371  6.9802985 7.315751  7.8271065]\n [9.202143  5.916399  6.0243287 5.724239 ]\n [5.8247943 7.4402814 5.6241655 6.778064 ]\n [8.610403  9.70216   5.1522646 8.826626 ]]\n<NDArray 4x4 @cpu(0)>\n\n[[8.734971  8.743319  5.4629807 9.518599 ]]\n<NDArray 1x4 @cpu(0)>\n"
 }
]
```

```{.python .input  n=86}
from mxnet import gluon
from mxnet import autograd
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
a = nd.random_normal(shape=(4, 4))
with autograd.record():
    y = net(a)
y.backward()
trainer.step(1)
```

```{.python .input  n=87}
print(net.collect_params())
```

```{.json .output n=87}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "sequential15_ (\n  Parameter sequential15_dense0_weight (shape=(4, 4), dtype=<class 'numpy.float32'>)\n  Parameter sequential15_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n  Parameter sequential15_dense2_weight (shape=(1, 4), dtype=<class 'numpy.float32'>)\n  Parameter sequential15_dense2_bias (shape=(1,), dtype=<class 'numpy.float32'>)\n)\n"
 }
]
```

```{.python .input  n=88}
print(net[0].weight.data())
print(net[1].weight.data())
print(net[2].weight.data())
```

```{.json .output n=88}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 46.127563 -44.180466 -52.159836 -98.03035 ]\n [ 41.710976 -41.90676  -50.33182  -93.24747 ]\n [ 35.595413 -28.681591 -35.441845 -67.93771 ]\n [ 37.813026 -38.277912 -52.37093  -90.49579 ]]\n<NDArray 4x4 @cpu(0)>\n\n[[ 46.127563 -44.180466 -52.159836 -98.03035 ]\n [ 41.710976 -41.90676  -50.33182  -93.24747 ]\n [ 35.595413 -28.681591 -35.441845 -67.93771 ]\n [ 37.813026 -38.277912 -52.37093  -90.49579 ]]\n<NDArray 4x4 @cpu(0)>\n\n[[-63.586525 -54.407845 -53.361164 -63.996933]]\n<NDArray 1x4 @cpu(0)>\n"
 }
]
```

```{.python .input  n=89}
print(net[0].weight.grad())
print(net[1].weight.grad())
print(net[2].weight.grad())
```

```{.json .output n=89}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[-376.0519   511.6076   594.75586 1058.5746 ]\n [-325.08832  478.23157  563.5615   989.71704]\n [-297.70618  361.21872  410.6601   747.1577 ]\n [-292.02625  479.80072  575.23193  993.2242 ]]\n<NDArray 4x4 @cpu(0)>\n\n[[-376.0519   511.6076   594.75586 1058.5746 ]\n [-325.08832  478.23157  563.5615   989.71704]\n [-297.70618  361.21872  410.6601   747.1577 ]\n [-292.02625  479.80072  575.23193  993.2242 ]]\n<NDArray 4x4 @cpu(0)>\n\n[[723.2149  631.51166 588.24146 735.15533]]\n<NDArray 1x4 @cpu(0)>\n"
 }
]
```
