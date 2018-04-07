# 使用GPU来计算

【注意】运行本教程需要GPU。没有GPU的同学可以大致理解下内容，至少是`context`这个概念，因为之后我们也会用到。但没有GPU不会影响运行之后的大部分教程（好吧，还是有点点，可能运行会稍微慢点）。

前面的教程里我们一直在使用CPU来计算，因为绝大部分的计算设备都有CPU。但CPU的设计目的是处理通用的计算，例如打开浏览器和运行Jupyter，它一般只有少数的一块区域复杂数值计算，例如`nd.dot(A, B)`。对于复杂的神经网络和大规模的数据来说，单块CPU可能不够给力。

常用的解决办法是要么使用多台机器来协同计算，要么使用数值计算更加强劲的硬件，或者两者一起使用。本教程关注使用单块Nvidia GPU来加速计算，更多的选项例如多GPU和多机器计算则留到后面。

首先需要确保至少有一块Nvidia显卡已经安装好了，然后下载安装显卡驱动和[CUDA](https://developer.nvidia.com/cuda-downloads)（推荐下载8.0，CUDA自带了驱动）。完成后应该可以通过`nvidia-smi`查看显卡信息了。（Windows用户需要设一下PATH：`set PATH=C:\Program Files\NVIDIA Corporation\NVSMI;%PATH%`）。

```{.python .input  n=1}
!nvidia-smi
```

```{.json .output n=1}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.\r\n\r\n"
 }
]
```

接下来要要确认正确安装了的`mxnet`的GPU版本。具体来说是卸载了`mxnet`（`pip uninstall mxnet`），然后根据CUDA版本安装`mxnet-cu75`或者`mxnet-cu80`（例如`pip install --pre mxnet-cu80`）。

使用pip来确认下：

```{.python .input  n=2}
import pip
for pkg in ['mxnet', 'mxnet-cu75', 'mxnet-cu91']:
    pip.main(['show', pkg])
```

```{.json .output n=2}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "You are using pip version 9.0.1, however version 9.0.3 is available.\nYou should consider upgrading via the 'pip install --upgrade pip' command.\nYou are using pip version 9.0.1, however version 9.0.3 is available.\nYou should consider upgrading via the 'pip install --upgrade pip' command.\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Name: mxnet-cu91\nVersion: 1.2.0b20180326\nSummary: MXNet is an ultra-scalable deep learning framework. This version uses CUDA-9.1.\nHome-page: https://github.com/apache/incubator-mxnet\nAuthor: UNKNOWN\nAuthor-email: UNKNOWN\nLicense: Apache 2.0\nLocation: /home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages\nRequires: graphviz, requests, numpy\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "You are using pip version 9.0.1, however version 9.0.3 is available.\nYou should consider upgrading via the 'pip install --upgrade pip' command.\n"
 }
]
```

## Context

MXNet使用Context来指定使用哪个设备来存储和计算。默认会将数据开在主内存，然后利用CPU来计算，这个由`mx.cpu()`来表示。GPU则由`mx.gpu()`来表示。注意`mx.cpu()`表示所有的物理CPU和内存，意味着计算上会尽量使用多有的CPU核。但`mx.gpu()`只代表一块显卡和其对应的显卡内存。如果有多块GPU，我们用`mx.gpu(i)`来表示第*i*块GPU（*i*从0开始）。

```{.python .input  n=3}
import mxnet as mx
[mx.cpu(), mx.gpu(), mx.gpu(1)]
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "[cpu(0), gpu(0), gpu(1)]"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## NDArray的GPU计算

每个NDArray都有一个`context`属性来表示它存在哪个设备上，默认会是`cpu`。这是为什么前面每次我们打印NDArray的时候都会看到`@cpu(0)`这个标识。

```{.python .input  n=4}
from mxnet import nd
x = nd.array([1,2,3])
x.context
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "cpu(0)"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### GPU上创建内存

我们可以在创建的时候指定创建在哪个设备上（如果GPU不能用或者没有装MXNet GPU版本，这里会有error）：

```{.python .input  n=5}
a = nd.array([1,2,3], ctx=mx.gpu())
b = nd.zeros((3,2), ctx=mx.gpu())
c = nd.random.uniform(shape=(2,3), ctx=mx.gpu())
(a,b,c)
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "(\n [1. 2. 3.]\n <NDArray 3 @gpu(0)>, \n [[0. 0.]\n  [0. 0.]\n  [0. 0.]]\n <NDArray 3x2 @gpu(0)>, \n [[0.6686509  0.17409194 0.3850025 ]\n  [0.24678314 0.35134333 0.8404298 ]]\n <NDArray 2x3 @gpu(0)>)"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

尝试将内存开到另外一块GPU上。如果不存在会报错。当然，如果你有大于10块GPU，那么下面代码会顺利执行。

```{.python .input  n=6}
import sys

try:
    nd.array([1,2,3], ctx=mx.gpu(10))
except mx.MXNetError as err:
    sys.stderr.write(str(err))
```

我们可以通过`copyto`和`as_in_context`来在设备直接传输数据。

```{.python .input  n=8}
y = x.copyto(mx.gpu())#x在cpu通过copyto传到gpu上面
z = x.as_in_context(mx.gpu())
(y, z)
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "(\n [1. 2. 3.]\n <NDArray 3 @gpu(0)>, \n [1. 2. 3.]\n <NDArray 3 @gpu(0)>)"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

这两个函数的主要区别是，如果源和目标的context一致，`as_in_context`不复制，而`copyto`总是会新建内存：

```{.python .input  n=9}
yy = y.as_in_context(mx.gpu())
zz = z.copyto(mx.gpu())
(yy is y, zz is z)
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "(True, False)"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### GPU上的计算

计算会在数据的`context`上执行。所以为了使用GPU，我们只需要事先将数据放在上面就行了。结果会自动保存在对应的设备上：

```{.python .input  n=10}
nd.exp(z + 2) * y
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "\n[ 20.085537 109.1963   445.2395  ]\n<NDArray 3 @gpu(0)>"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

注意所有计算要求输入数据在同一个设备上。不一致的时候系统不进行自动复制。这个设计的目的是因为设备之间的数据交互通常比较昂贵，我们希望用户确切的知道数据放在哪里，而不是隐藏这个细节。下面代码尝试将CPU上`x`和GPU上的`y`做运算。

```{.python .input  n=12}
import sys
try:
    x + y
except mx.MXNetError as err:
    sys.stderr.write(str(err))#x在cpu,y在gpu
```

```{.json .output n=12}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "[09:58:18] src/imperative/./imperative_utils.h:55: Check failed: inputs[i]->ctx().dev_mask() == ctx.dev_mask() (2 vs. 1) Operator broadcast_add require all inputs live on the same context. But the first argument is on cpu(0) while the 2-th argument is on gpu(0)\n\nStack trace returned 10 entries:\n[bt] (0) /home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2e9c82) [0x7f1471baec82]\n[bt] (1) /home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2ea258) [0x7f1471baf258]\n[bt] (2) /home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x273712d) [0x7f1473ffc12d]\n[bt] (3) /home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2741061) [0x7f1474006061]\n[bt] (4) /home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x26838cb) [0x7f1473f488cb]\n[bt] (5) /home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(MXImperativeInvokeEx+0x6f) [0x7f1473f48e8f]\n[bt] (6) /home/zhang/miniconda3/envs/gluon/lib/python3.6/lib-dynload/../../libffi.so.6(ffi_call_unix64+0x4c) [0x7f14b858fec0]\n[bt] (7) /home/zhang/miniconda3/envs/gluon/lib/python3.6/lib-dynload/../../libffi.so.6(ffi_call+0x22d) [0x7f14b858f87d]\n[bt] (8) /home/zhang/miniconda3/envs/gluon/lib/python3.6/lib-dynload/_ctypes.cpython-36m-x86_64-linux-gnu.so(_ctypes_callproc+0x2ce) [0x7f14b87a4dee]\n[bt] (9) /home/zhang/miniconda3/envs/gluon/lib/python3.6/lib-dynload/_ctypes.cpython-36m-x86_64-linux-gnu.so(+0x12825) [0x7f14b87a5825]\n\n"
 }
]
```

### 默认会复制回CPU的操作

如果某个操作需要将NDArray里面的内容转出来，例如打印或变成numpy格式，如果需要的话系统都会自动将数据copy到主内存。

```{.python .input  n=11}
print(y)
print(y.asnumpy())#走cpu
print(y.sum().asscalar())#走cpu
```

## Gluon的GPU计算

同NDArray类似，Gluon的大部分函数可以通过`ctx`指定设备。下面代码将模型参数初始化在GPU上：

```{.python .input  n=12}
from mxnet import gluon
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))

net.initialize(ctx=mx.gpu())#默认是cpu
```

输入GPU上的数据，会在GPU上计算结果

```{.python .input  n=13}
data = nd.random.uniform(shape=[3,2], ctx=mx.gpu())#生成在gpu上的矩阵
net(data)
```

确认下权重：

```{.python .input  n=14}
net[0].weight.data()
```

## 总结

通过`context`我们可以很容易在不同的设备上计算。

## 练习

- 试试大一点的计算任务，例如大矩阵的乘法，看看CPU和GPU的速度区别。如果是计算量很小的任务呢？
- 试试CPU和GPU之间传递数据的速度
- GPU上如何读写模型呢？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/988)
