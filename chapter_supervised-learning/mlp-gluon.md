# 多层感知机 --- 使用Gluon

我们只需要稍微改动[多类Logistic回归](../chapter_crashcourse/softmax-regression-gluon.md)来实现多层感知机。

## 定义模型

唯一的区别在这里，我们加了一行进来。

```{.python .input  n=5}
from mxnet import gluon

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(256, activation="relu"))#中间层维度
    net.add(gluon.nn.Dense(10))#输出层维度
net.initialize()
```

## 读取数据并训练

```{.python .input  n=8}
import sys
sys.path.append('..')
from mxnet import ndarray as nd
from mxnet import autograd as ag
import utils

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate': 0.5})#更新参数

for epoch in range(5):
    train_loss = 0
    train_acc = 0
    for data, label in train_data:
        with ag.record():
            output = net(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        trainer.step(batch_size)
        
        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output,label)
    
    test_acc =utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

```{.json .output n=8}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:84: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  label = np.fromstring(fin.read(), dtype=np.uint8).astype(np.int32)\n/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:88: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  data = np.fromstring(fin.read(), dtype=np.uint8)\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.473787, Train acc 0.824536, Test acc 0.844451\nEpoch 1. Loss: 0.415110, Train acc 0.845903, Test acc 0.837240\nEpoch 2. Loss: 0.379329, Train acc 0.860644, Test acc 0.859475\nEpoch 3. Loss: 0.357336, Train acc 0.867839, Test acc 0.875000\nEpoch 4. Loss: 0.338786, Train acc 0.875701, Test acc 0.861779\n"
 }
]
```

## 结论

通过Gluon我们可以更方便地构造多层神经网络。

## 练习

- 尝试多加入几个隐含层，对比从0开始的实现。
- 尝试使用一个另外的激活函数，可以使用`help(nd.Activation)`或者[线上文档](https://mxnet.apache.org/api/python/ndarray.html#mxnet.ndarray.Activation)查看提供的选项。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/738)

```{.python .input  n=2}
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(256, activation="softrelu"))#中间层维度
    net.add(gluon.nn.Dense(10))#输出层维度
net.initialize()
```

```{.python .input  n=5}
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate': 0.5})#更新参数

for epoch in range(5):
    train_loss = 0
    train_acc = 0
    for data, label in train_data:
        with ag.record():
            output = net(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        trainer.step(batch_size)
        
        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output,label)
    
    test_acc =utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

```{.json .output n=5}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:84: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  label = np.fromstring(fin.read(), dtype=np.uint8).astype(np.int32)\n/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:88: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  data = np.fromstring(fin.read(), dtype=np.uint8)\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.927928, Train acc 0.686365, Test acc 0.787260\nEpoch 1. Loss: 0.507367, Train acc 0.812116, Test acc 0.794972\nEpoch 2. Loss: 0.455969, Train acc 0.832716, Test acc 0.846554\nEpoch 3. Loss: 0.422504, Train acc 0.845586, Test acc 0.821715\nEpoch 4. Loss: 0.400679, Train acc 0.853983, Test acc 0.859175\n"
 }
]
```

```{.python .input  n=6}
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(256, activation="tanh"))#中间层维度
    net.add(gluon.nn.Dense(10))#输出层维度
net.initialize()
```

```{.python .input  n=7}
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate': 0.5})#更新参数

for epoch in range(5):
    train_loss = 0
    train_acc = 0
    for data, label in train_data:
        with ag.record():
            output = net(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        trainer.step(batch_size)
        
        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output,label)
    
    test_acc =utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

```{.json .output n=7}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:84: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  label = np.fromstring(fin.read(), dtype=np.uint8).astype(np.int32)\n/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:88: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  data = np.fromstring(fin.read(), dtype=np.uint8)\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 3.532300, Train acc 0.596605, Test acc 0.775841\nEpoch 1. Loss: 0.806639, Train acc 0.763822, Test acc 0.841747\nEpoch 2. Loss: 0.595257, Train acc 0.802501, Test acc 0.830429\nEpoch 3. Loss: 0.512589, Train acc 0.821798, Test acc 0.848257\nEpoch 4. Loss: 0.460213, Train acc 0.835337, Test acc 0.855769\n"
 }
]
```

```{.python .input  n=9}
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(256, activation="sigmoid"))#中间层维度
    net.add(gluon.nn.Dense(10))#输出层维度
net.initialize()
```

```{.python .input  n=10}
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate': 0.5})#更新参数

for epoch in range(5):
    train_loss = 0
    train_acc = 0
    for data, label in train_data:
        with ag.record():
            output = net(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        trainer.step(batch_size)
        
        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output,label)
    
    test_acc =utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

```{.json .output n=10}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:84: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  label = np.fromstring(fin.read(), dtype=np.uint8).astype(np.int32)\n/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:88: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  data = np.fromstring(fin.read(), dtype=np.uint8)\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.884759, Train acc 0.678953, Test acc 0.784856\nEpoch 1. Loss: 0.541637, Train acc 0.801749, Test acc 0.815505\nEpoch 2. Loss: 0.488290, Train acc 0.824102, Test acc 0.840445\nEpoch 3. Loss: 0.457048, Train acc 0.835570, Test acc 0.837740\nEpoch 4. Loss: 0.435448, Train acc 0.843450, Test acc 0.843950\n"
 }
]
```

```{.python .input  n=15}
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(400, activation="relu"))#中间层维度
    net.add(gluon.nn.Dense(400, activation="relu"))
    net.add(gluon.nn.Dense(10))#输出层维度
net.initialize()
```

```{.python .input  n=16}
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate': 0.1})#更新参数

for epoch in range(20):
    train_loss = 0
    train_acc = 0
    for data, label in train_data:
        with ag.record():
            output = net(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        trainer.step(batch_size)
        
        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output,label)
    
    test_acc =utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

```{.json .output n=16}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:84: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  label = np.fromstring(fin.read(), dtype=np.uint8).astype(np.int32)\n/home/zhang/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:88: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  data = np.fromstring(fin.read(), dtype=np.uint8)\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.831980, Train acc 0.720353, Test acc 0.799980\nEpoch 1. Loss: 0.522763, Train acc 0.816039, Test acc 0.823718\nEpoch 2. Loss: 0.463750, Train acc 0.834285, Test acc 0.845052\nEpoch 3. Loss: 0.429519, Train acc 0.847256, Test acc 0.854667\nEpoch 4. Loss: 0.404491, Train acc 0.855736, Test acc 0.858073\nEpoch 5. Loss: 0.384050, Train acc 0.861912, Test acc 0.868289\nEpoch 6. Loss: 0.371470, Train acc 0.867121, Test acc 0.871494\nEpoch 7. Loss: 0.355875, Train acc 0.871277, Test acc 0.873197\nEpoch 8. Loss: 0.345555, Train acc 0.874800, Test acc 0.871294\nEpoch 9. Loss: 0.335227, Train acc 0.878222, Test acc 0.877103\nEpoch 10. Loss: 0.325728, Train acc 0.881878, Test acc 0.876603\nEpoch 11. Loss: 0.318732, Train acc 0.884131, Test acc 0.876903\nEpoch 12. Loss: 0.310465, Train acc 0.887186, Test acc 0.878305\nEpoch 13. Loss: 0.302249, Train acc 0.889857, Test acc 0.882812\nEpoch 14. Loss: 0.296963, Train acc 0.892361, Test acc 0.876502\nEpoch 15. Loss: 0.291015, Train acc 0.893263, Test acc 0.880909\nEpoch 16. Loss: 0.283960, Train acc 0.896384, Test acc 0.887921\nEpoch 17. Loss: 0.281534, Train acc 0.897035, Test acc 0.887520\nEpoch 18. Loss: 0.274352, Train acc 0.898988, Test acc 0.887821\nEpoch 19. Loss: 0.268384, Train acc 0.901759, Test acc 0.887019\n"
 }
]
```
