# VGG：使用重复元素的非常深的网络

我们从Alexnet看到网络的层数的激增。这个意味着即使是用Gluon手动写代码一层一层的堆每一层也很麻烦，更不用说从0开始了。幸运的是编程语言提供了很好的方法来解决这个问题：函数和循环。如果网络结构里面有大量重复结构，那么我们可以很紧凑来构造这些网络。第一个使用这种结构的深度网络是VGG。

## VGG架构

VGG的一个关键是使用很多有着相对小的kernel（$3\times 3$）的卷积层然后接上一个池化层，之后再将这个模块重复多次。下面我们先定义一个这样的块：

```{.python .input}
from mxnet.gluon import nn

def vgg_block(num_convs, channels):
    out = nn.Sequential()
    for _ in range(num_convs):
        out.add(
            nn.Conv2D(channels=channels, kernel_size=3,
                      padding=1, activation='relu')
        )
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out
```

```{.python .input  n=5}
from mxnet.gluon import nn

def vgg_block(num_convs, channels):
    out = nn.Sequential()
    for _ in range(num_convs):
        out.add(
            nn.Conv2D(channels = channels, kernel_size=3,
                     padding=1,activation='relu')
        )
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out
```

我们实例化一个这样的块，里面有两个卷积层，每个卷积层输出通道是128：

```{.python .input  n=6}
from mxnet import nd

blk = vgg_block(2, 128)
blk.initialize()
x = nd.random.uniform(shape=(2,3,16,16))
y = blk(x)
y.shape
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "(2, 128, 8, 8)"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input}
from mxnet import nd

blk = vgg_block(2, 128)
blk.initialize()
x = nd.random.uniform(shape=(2,3,16,16))
y = blk(x)
y.shape
```

```{.python .input  n=8}
k = nd.random.uniform(shape=(2,3,3,3))
k
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "\n[[[[0.57432526 0.03362509 0.6532008 ]\n   [0.9689618  0.65210325 0.32099724]\n   [0.43141845 0.22126268 0.8965466 ]]\n\n  [[0.1412639  0.36756188 0.09725992]\n   [0.43586493 0.9840422  0.89192337]\n   [0.26034093 0.806194   0.53702253]]\n\n  [[0.7038886  0.44792616 0.10022689]\n   [0.09956908 0.9194826  0.35231167]\n   [0.7142413  0.46924916 0.998847  ]]]\n\n\n [[[0.84114015 0.1494483  0.90464777]\n   [0.86812603 0.03755938 0.16249293]\n   [0.50831544 0.6155596  0.16684751]]\n\n  [[0.12381998 0.779051   0.8480082 ]\n   [0.8649333  0.807319   0.4113967 ]\n   [0.56910074 0.13997258 0.4071833 ]]\n\n  [[0.03322238 0.06916699 0.98257494]\n   [0.69742876 0.37329075 0.45354268]\n   [0.42007536 0.7220556  0.05058811]]]]\n<NDArray 2x3x3x3 @cpu(0)>"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

可以看到经过一个这样的块后，长宽会减半，通道也会改变。

然后我们定义如何将这些块堆起来：

```{.python .input}
def vgg_stack(architecture):
    out = nn.Sequential()
    for (num_convs, channels) in architecture:
        out.add(vgg_block(num_convs, channels))
    return out
```

```{.python .input}
def vgg_stack(architecture):
    out = nn.Sequential()
    for (num_convs, channels)in architecture:
        out.add(vgg_block(num_convs, channels))
    return out
```

这里我们定义一个最简单的一个VGG结构，它有8个卷积层，和跟Alexnet一样的3个全连接层。这个网络又称VGG 11.

```{.python .input}
num_outputs = 10
architecture = ((1,64), (1,128), (2,256), (2,512), (2,512))
net = nn.Sequential()
# add name_scope on the outermost Sequential
with net.name_scope():
    net.add(
        vgg_stack(architecture),
        nn.Flatten(),
        nn.Dense(4096, activation="relu"),
        nn.Dropout(.5),
        nn.Dense(4096, activation="relu"),
        nn.Dropout(.5),
        nn.Dense(num_outputs))
```

```{.python .input}
num_outputs = 10
architecture = ((1,64),(1,128),(2,256),(2,512),(2,512))
net.Sequential()
with net.name_scope():
    net.add(
        vgg_stack(architecture),
        nn.Flatten(),
        nn.Dense(4049, activation = "relu"),
        nn.Dropout(.5)
        nn.Dense(4096,activation = "relu")
        nn.Dropout(.5),
        nn.Dense(num_outputs)
    )
```

## 模型训练

这里跟Alexnet的训练代码一样除了我们只将图片扩大到$96\times 96$来节省些计算，和默认使用稍微大点的学习率。

```{.python .input}
import sys
sys.path.append('..')
import utils
from mxnet import gluon
from mxnet import init

train_data, test_data = utils.load_data_fashion_mnist(
    batch_size=64, resize=96)

ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 
                        'sgd', {'learning_rate': 0.05})
utils.train(train_data, test_data, net, loss,
            trainer, ctx, num_epochs=1)
```

## 总结

通过使用重复的元素，我们可以通过循环和函数来定义模型。使用不同的配置(`architecture`)可以得到一系列不同的模型。


## 练习

- 尝试多跑几轮，看看跟LeNet/Alexnet比怎么样？
- 尝试下构造VGG其他常用模型，例如VGG16， VGG19. （提示：可以参考[VGG论文](https://arxiv.org/abs/1409.1556)里的表1。）
- 把图片从默认的$224\times 224$降到$96\times 96$有什么影响？


**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/1277)
