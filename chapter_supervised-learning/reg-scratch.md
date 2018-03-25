# 正则化 --- 从0开始

本章从0开始介绍如何的正则化来应对[过拟合](underfit-overfit.md)问题。

## 高维线性回归

我们使用高维线性回归为例来引入一个过拟合问题。


具体来说我们使用如下的线性函数来生成每一个数据样本

$$y = 0.05 + \sum_{i = 1}^p 0.01x_i +  \text{noise}$$

这里噪音服从均值0和标准差为0.01的正态分布。

需要注意的是，我们用以上相同的数据生成函数来生成训练数据集和测试数据集。为了观察过拟合，我们特意把训练数据样本数设低，例如$n=20$，同时把维度升高，例如$p=200$。

```{.python .input  n=1}
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import mxnet as mx

num_train = 20#训练集样本量
num_test = 100#测试集样本量
num_inputs = 200#输入维度？
```

## 生成数据集


这里定义模型真实参数。

```{.python .input  n=13}
true_w = nd.ones((num_inputs, 1)) * 0.01#一列
true_b = 0.05
```

我们接着生成训练和测试数据集。

```{.python .input  n=3}
X = nd.random.normal(shape=(num_train + num_test, num_inputs))#行是样本量，列是属性
y = nd.dot(X, true_w) + true_b
y += .01 * nd.random.normal(shape=y.shape)#一列

X_train, X_test = X[:num_train, :], X[num_train:, :]
y_train, y_test = y[:num_train], y[num_train:]
```

当我们开始训练神经网络的时候，我们需要不断读取数据块。这里我们定义一个函数它每次返回`batch_size`个随机的样本和对应的目标。我们通过python的`yield`来构造一个迭代器。

```{.python .input  n=11}
import random
batch_size = 1
def data_iter(num_examples):
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i: min(i+batch_size, num_examples)])
        yield X.take(j), y.take(j)
```

## 初始化模型参数

下面我们随机初始化模型参数。之后训练时我们需要对这些参数求导来更新它们的值，所以我们需要创建它们的梯度。

```{.python .input  n=5}
def init_params():
    w = nd.random_normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    for param in params:
        param.attach_grad()
    return params
```

## $L_2$范数正则化

这里我们引入$L_2$范数正则化。不同于在训练时仅仅最小化损失函数(Loss)，我们在训练时其实在最小化

$$\text{loss} + \lambda \sum_{p \in \textrm{params}}\|p\|_2^2。$$

直观上，$L_2$范数正则化试图惩罚较大绝对值的参数值。下面我们定义L2正则化。注意有些时候大家对偏移加罚，有时候不加罚。通常结果上两者区别不大。这里我们演示对偏移也加罚的情况：

```{.python .input  n=6}
def L2_penalty(w, b):
    return ((w**2).sum() + b**2) / 2
```

## 定义训练和测试

下面我们定义剩下的所需要的函数。这个跟之前的教程大致一样，主要是区别在于计算`loss`的时候我们加上了L2正则化，以及我们将训练和测试损失都画了出来。

```{.python .input  n=7}
%matplotlib inline
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt
import numpy as np

def net(X, w, b):
    return nd.dot(X, w) + b

def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):#params = [w,b]
    for param in params:
        param[:] = param - lr * param.grad / batch_size#加入正则
        
def test(net, params, X, y):
    return square_loss(net(X, *params), y).mean().asscalar()
    #return np.mean(square_loss(net(X, *params), y).asnumpy())
    #params = [w,b]

def train(lambd):
    epochs = 10
    learning_rate = 0.005
    w, b = params = init_params()
    train_loss = []
    test_loss = []
    for e in range(epochs):        
        for data, label in data_iter(num_train):
            with autograd.record():
                output = net(data, *params)
                loss = square_loss(
                    output, label) + lambd * L2_penalty(*params)
            loss.backward()
            sgd(params, learning_rate, batch_size)
        train_loss.append(test(net, params, X_train, y_train))
        test_loss.append(test(net, params, X_test, y_test))
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    plt.show()
    return 'learned w[:10]:', w[:10].T, 'learned b:', b
```

## 观察过拟合

接下来我们训练并测试我们的高维线性回归模型。注意这时我们并未使用正则化。

```{.python .input  n=12}
train(0)
```

```{.json .output n=12}
[
 {
  "data": {
   "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXQAAAD8CAYAAABn919SAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvhp/UCwAAFMlJREFUeJzt3X+MXeV95/H31zNjxgbzwz/X2DjjFEqh+WHIQKFGVRKXgEMXHKVBSUrlVkjOH9ku7TZp7ErJLtJKpdKKkkgNkQO0XiVLQg3IbEJTBxeUkBDIGEhjsNsxhOCJHXts4oDBBtt8+8ccm7GZ8dyZuXfuzOP3Sxqdc55zzj3fe+z53Geee+49kZlIkia+Sc0uQJJUHwa6JBXCQJekQhjoklQIA12SCmGgS1IhDHRJKoSBLkmFMNAlqRCtY3mwmTNnZkdHx1geUpImvI0bN+7OzFlDbTemgd7R0UFXV9dYHlKSJryI+Hkt2znkIkmFMNAlqRAGuiQVwkCXpEIY6JJUCANdkgphoEtSIcb0OvQR+8k34KWfQetkaDkFWia/Nd86uW/56PxQ66ufiGY/K0mqq4kR6Jvug+5/qe9jHgn5ljZoPfIicMpbgT/QfOsp1c8UaGuHtqnQ2g5tU96a9p9vrbZpaz92n5a2+j4XSWKiBPof3QNvvgmH34DDr8Phg3Do9b75Q29U7W+81XZ0/RtvTWuZP1Tte/j1vvlDB+D1l6tjVMc6dKDv5+B+ePPgyJ5PtAzvRaC12qZ/W4xytKzeNwc/+hdPDLxcyzY1Lw/WNgp1OR/ecF0n8JtXQfsZDT3ExAh0gEmTYFJ7X5iNF4cPwaH9cPBANa1+jgT+oQNw8LV+64+07z+u7bVj99n/UrX+uMfJN5v9jCWN1Kd/bKCPay2t0DINTpnW+GNlVn95VC8CdekN1ut9hKqWo73c45dr2aaWx2DwberSU6/DY/jejAZzxjkNP8SQgR4R5wPf7Nf0TuALwP+t2juAF4DrM/NX9S9RQF9QtFZv9jb4VV7SxDTkQGxm/ntmLsrMRcD7gNeA+4GVwIbMPA/YUC1LkppkuO+sLQGey8yfA9cBa6r2NcCyehYmSRqe4Qb6x4G7q/k5mbkDoJrOrmdhkqThqTnQI2IycC3wT8M5QESsiIiuiOjq7e0dbn2SpBoNp4e+FHgyM3dWyzsjYi5ANd010E6ZuTozOzOzc9asIe+gJEkaoeEE+id4a7gF4AFgeTW/HFhXr6IkScNXU6BHxFTgSuC+fs23AFdGRHe17pb6lydJqlVNHyzKzNeAGce17aHvqhdJ0jjg1+dKUiEMdEkqhIEuSYUw0CWpEAa6JBXCQJekQhjoklQIA12SCmGgS1IhDHRJKoSBLkmFMNAlqRAGuiQVwkCXpEIY6JJUCANdkgphoEtSIQx0SSqEgS5Jhaj1JtFnRsTaiNgSEZsj4vKImB4R342I7mp6VqOLlSQNrtYe+heB72TmbwHvBTYDK4ENmXkesKFaliQ1yZCBHhGnA78H3AmQmW9k5l7gOmBNtdkaYFmjipQkDa2WHvo7gV7gHyLiqYi4IyJOBeZk5g6Aajp7oJ0jYkVEdEVEV29vb90KlyQdq5ZAbwUuBm7PzIuAVxnG8Epmrs7MzszsnDVr1gjLlCQNpZZA7wF6MvPxanktfQG/MyLmAlTTXY0pUZJUiyEDPTN/CWyLiPOrpiXAs8ADwPKqbTmwriEVSpJq0lrjdn8GfD0iJgPPA39K34vBPRFxI/Ai8LHGlChJqkVNgZ6ZTwOdA6xaUt9yJEkj5SdFJakQBrokFcJAl6RCGOiSVAgDXZIKYaBLUiEMdEkqhIEuSYUw0CWpEAa6JBXCQJekQhjoklQIA12SCmGgS1IhDHRJKoSBLkmFMNAlqRAGuiQVoqZb0EXEC8ArwGHgUGZ2RsR04JtAB/ACcH1m/qoxZUqShjKcHvoHMnNRZh65t+hKYENmngdsqJYlSU0ymiGX64A11fwaYNnoy5EkjVStgZ7A+ojYGBErqrY5mbkDoJrOHmjHiFgREV0R0dXb2zv6iiVJA6ppDB1YnJnbI2I28N2I2FLrATJzNbAaoLOzM0dQoySpBjX10DNzezXdBdwPXArsjIi5ANV0V6OKlCQNbchAj4hTI2LakXngQ8Am4AFgebXZcmBdo4qUJA2tliGXOcD9EXFk+/+Xmd+JiB8D90TEjcCLwMcaV6YkaShDBnpmPg+8d4D2PcCSRhQlSRo+PykqSYUw0CWpEAa6JBXCQJekQhjoklQIA12SCmGgS1IhDHRJKoSBLkmFMNAlqRAGuiQVwkCXpEIY6JJUCANdkgphoEtSIWq9p6gkNcXBgwfp6enhwIEDzS6l4drb25k/fz5tbW0j2t9AlzSu9fT0MG3aNDo6OqjunFakzGTPnj309PSwcOHCET2GQy6SxrUDBw4wY8aMosMcICKYMWPGqP4SqTnQI6IlIp6KiG9Vywsj4vGI6I6Ib0bE5BFXIUknUHqYHzHa5zmcHvpNwOZ+y38L/F1mngf8CrhxVJVI0ji0d+9evvzlLw97vw9/+MPs3bu3ARUNrqZAj4j5wDXAHdVyAB8E1labrAGWNaJASWqmwQL98OHDJ9zvwQcf5Mwzz2xUWQOq9U3R24C/AqZVyzOAvZl5qFruAeYNtGNErABWACxYsGDklUpSE6xcuZLnnnuORYsW0dbWxmmnncbcuXN5+umnefbZZ1m2bBnbtm3jwIED3HTTTaxYsQKAjo4Ourq62LdvH0uXLuWKK67ghz/8IfPmzWPdunVMmTKl7rUOGegR8QfArszcGBHvP9I8wKY50P6ZuRpYDdDZ2TngNpJUi5v//zM8u/3luj7mhWefzv/8r7896PpbbrmFTZs28fTTT/PII49wzTXXsGnTpqNXotx1111Mnz6d/fv3c8kll/DRj36UGTNmHPMY3d3d3H333Xz1q1/l+uuv59577+WGG26o6/OA2nroi4FrI+LDQDtwOn099jMjorXqpc8Htte9OkkaZy699NJjLiv80pe+xP333w/Atm3b6O7uflugL1y4kEWLFgHwvve9jxdeeKEhtQ0Z6Jm5ClgFUPXQP5OZfxQR/wT8IfANYDmwriEVSlLlRD3psXLqqacenX/kkUd46KGHeOyxx5g6dSrvf//7B7zs8JRTTjk639LSwv79+xtS22iuQ/8c8D8iYit9Y+p31qckSRo/pk2bxiuvvDLgul//+tecddZZTJ06lS1btvCjH/1ojKs71rA+KZqZjwCPVPPPA5fWvyRJGj9mzJjB4sWLede73sWUKVOYM2fO0XVXX301X/nKV3jPe97D+eefz2WXXdbESiEyx+59ys7Ozuzq6hqz40ma+DZv3swFF1zQ7DLGzEDPNyI2ZmbnUPv60X9JKoSBLkmFMNAlqRAGuiQVwkCXpEIY6JJUCANdkk5gpF+fC3Dbbbfx2muv1bmiwRnoknQCEynQvaeoJJ1A/6/PvfLKK5k9ezb33HMPr7/+Oh/5yEe4+eabefXVV7n++uvp6enh8OHDfP7zn2fnzp1s376dD3zgA8ycOZOHH3644bUa6JImjn9eCb/8aX0f87+8G5beMujq/l+fu379etauXcsTTzxBZnLttdfyve99j97eXs4++2y+/e1vA33f8XLGGWdw66238vDDDzNz5sz61jwIh1wkqUbr169n/fr1XHTRRVx88cVs2bKF7u5u3v3ud/PQQw/xuc99ju9///ucccYZTanPHrqkieMEPemxkJmsWrWKT33qU29bt3HjRh588EFWrVrFhz70Ib7whS+MeX320CXpBPp/fe5VV13FXXfdxb59+wD4xS9+wa5du9i+fTtTp07lhhtu4DOf+QxPPvnk2/YdC/bQJekE+n997tKlS/nkJz/J5ZdfDsBpp53G1772NbZu3cpnP/tZJk2aRFtbG7fffjsAK1asYOnSpcydO3dM3hT163MljWt+fa5fnytJJx0DXZIKMWSgR0R7RDwRET+JiGci4uaqfWFEPB4R3RHxzYiY3PhyJUmDqaWH/jrwwcx8L7AIuDoiLgP+Fvi7zDwP+BVwY+PKlHQyG8v3+ppptM9zyEDPPvuqxbbqJ4EPAmur9jXAslFVIkkDaG9vZ8+ePcWHemayZ88e2tvbR/wYNV22GBEtwEbgXODvgeeAvZl5qNqkB5g34iokaRDz58+np6eH3t7eZpfScO3t7cyfP3/E+9cU6Jl5GFgUEWcC9wMDXUM04MtnRKwAVgAsWLBghGVKOlm1tbWxcOHCZpcxIQzrKpfM3As8AlwGnBkRR14Q5gPbB9lndWZ2ZmbnrFmzRlOrJOkEarnKZVbVMycipgC/D2wGHgb+sNpsObCuUUVKkoZWy5DLXGBNNY4+CbgnM78VEc8C34iI/w08BdzZwDolSUMYMtAz89+AiwZofx64tBFFSZKGz0+KSlIhDHRJKoSBLkmFMNAlqRAGuiQVwkCXpEIY6JJUCANdkgphoEtSIQx0SSqEgS5JhTDQJakQBrokFcJAl6RCGOiSVAgDXZIKYaBLUiEMdEkqRC03iT4nIh6OiM0R8UxE3FS1T4+I70ZEdzU9q/HlSpIGU0sP/RDwl5l5AXAZ8OmIuBBYCWzIzPOADdWyJKlJhgz0zNyRmU9W868Am4F5wHXAmmqzNcCyRhUpSRrasMbQI6IDuAh4HJiTmTugL/SB2fUuTpJUu5oDPSJOA+4F/jwzXx7Gfisioisiunp7e0dSoySpBjUFekS00RfmX8/M+6rmnRExt1o/F9g10L6ZuTozOzOzc9asWfWoWZI0gFqucgngTmBzZt7ab9UDwPJqfjmwrv7lSZJq1VrDNouBPwZ+GhFPV21/DdwC3BMRNwIvAh9rTImSpFoMGeiZ+SgQg6xeUt9yJEkj5SdFJakQBrokFcJAl6RCGOiSVAgDXZIKYaBLUiEMdEkqhIEuSYUw0CWpEAa6JBXCQJekQhjoklQIA12SCmGgS1IhDHRJKoSBLkmFMNAlqRAGuiQVopabRN8VEbsiYlO/tukR8d2I6K6mZzW2TEnSUGrpof8jcPVxbSuBDZl5HrChWpYkNdGQgZ6Z3wNeOq75OmBNNb8GWFbnuiRJwzTSMfQ5mbkDoJrOrl9JkqSRaPibohGxIiK6IqKrt7e30YeTpJPWSAN9Z0TMBaimuwbbMDNXZ2ZnZnbOmjVrhIeTJA1lpIH+ALC8ml8OrKtPOZKkkarlssW7gceA8yOiJyJuBG4BroyIbuDKalmS1EStQ22QmZ8YZNWSOtciSRoFPykqSYUw0CWpEAa6JBXCQJekQhjoklQIA12SCmGgS1IhDHRJKoSBLkmFMNAlqRAGuiQVYsjvchkPfrB1NwFc/I6zaG9raXY5kjQuTYhA/+KGbp742Uu0t03iko7pXHHuTBafO5ML557OpEnR7PIkaVyYEIF+159cwuPP7+HRrbv5wdbd/M0/bwFg+qmT+d3fmHE04M+ZPrXJlUpS80yIQD/tlFaWXDCHJRfMAWDnywf4wdbdPLp1N4927+Zb/7YDgHfMmMoV587kinNncvlvzODMqZObWbYkjanIzDE7WGdnZ3Z1ddX1MTOTrbv2He29P/bcHl594zAR8J55Z7C4CnjH3yVNVBGxMTM7h9xuogf68Q4efpOfbNt7NOCfenEvh95Mx98lTVgnbaAfb9/rh46Ovz/avZvuXfsAx98lTRy1BvqoxtAj4mrgi0ALcEdmjrt7iw46/t7dNwbv+LukUoy4hx4RLcB/0HeT6B7gx8AnMvPZwfZpRg/9RPqPvz/avZsfPe/4u6Txp+FDLhFxOfC/MvOqankVQGb+zWD7jLdAP96R8ffvd1fj79v2cvi48fdzpk+lZVLQEtE3HehnoHX92lonBZOOtLUct32EY/uSjjEWQy7zgG39lnuA3xnF4zVdW8skOjum09kxnb+48jd55cBBnvjZS0cD/sj1740WQV/oH/fC0L9tUgwc+v2bj5knBt/umPYYsJ0atm+m8VGFNLg7l1/CghmNfa9uNIE+0O/Q27r7EbECWAGwYMGCURxu7E1rbztm/H3XKwd46dU3OPxmDvyTo2w7XPv2R050/z+wkmMWBpqt9skB1+Ug+wy2/dv/tZsjx0sh0glMbm38V2eNJtB7gHP6Lc8Hth+/UWauBlZD35DLKI7XdLOntTN7Wnuzy5CkAY3mJePHwHkRsTAiJgMfBx6oT1mSpOEacQ89Mw9FxH8D/oW+yxbvysxn6laZJGlYRnUdemY+CDxYp1okSaPgDS4kqRAGuiQVwkCXpEIY6JJUCANdkgoxpl+fGxG9wM9HuPtMYHcdy5noPB9v8Vwcy/NxrBLOxzsyc9ZQG41poI9GRHTV8uU0JwvPx1s8F8fyfBzrZDofDrlIUiEMdEkqxEQK9NXNLmCc8Xy8xXNxLM/HsU6a8zFhxtAlSSc2kXrokqQTmBCBHhFXR8S/R8TWiFjZ7HqaJSLOiYiHI2JzRDwTETc1u6bxICJaIuKpiPhWs2tptog4MyLWRsSW6v/J5c2uqVki4i+q35NNEXF3RBR/M4NxH+jVzaj/HlgKXAh8IiIubG5VTXMI+MvMvAC4DPj0SXwu+rsJ2NzsIsaJLwLfyczfAt7LSXpeImIe8N+Bzsx8F31f8f3x5lbVeOM+0IFLga2Z+XxmvgF8A7iuyTU1RWbuyMwnq/lX6PtlndfcqporIuYD1wB3NLuWZouI04HfA+4EyMw3MnNvc6tqqlZgSkS0AlMZ4I5qpZkIgT7QzahP6hADiIgO4CLg8eZW0nS3AX8FvNnsQsaBdwK9wD9UQ1B3RMSpzS6qGTLzF8D/AV4EdgC/zsz1za2q8SZCoNd0M+qTSUScBtwL/HlmvtzsepolIv4A2JWZG5tdyzjRClwM3J6ZFwGvAifle04RcRZ9f8kvBM4GTo2IG5pbVeNNhECv6WbUJ4uIaKMvzL+emfc1u54mWwxcGxEv0DcU98GI+FpzS2qqHqAnM4/81baWvoA/Gf0+8LPM7M3Mg8B9wO82uaaGmwiB7s2oKxER9I2Pbs7MW5tdT7Nl5qrMnJ+ZHfT9v/jXzCy+FzaYzPwlsC0izq+algDPNrGkZnoRuCwipla/N0s4Cd4gHtU9RceCN6M+xmLgj4GfRsTTVdtfV/d2lQD+DPh61fl5HvjTJtfTFJn5eESsBZ6k7+qwpzgJPjHqJ0UlqRATYchFklQDA12SCmGgS1IhDHRJKoSBLkmFMNAlqRAGuiQVwkCXpEL8J1rzsuK0ibdjAAAAAElFTkSuQmCC\n",
   "text/plain": "<Figure size 432x288 with 1 Axes>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "data": {
   "text/plain": "('learned w[:10]:', \n [[ 0.5558106   0.8642054  -0.45820233  1.527333    0.7745064   0.09938803\n    1.5138943  -0.11311878 -1.3944883  -1.2200681 ]]\n <NDArray 1x10 @cpu(0)>, 'learned b:', \n [0.29741755]\n <NDArray 1 @cpu(0)>)"
  },
  "execution_count": 12,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

即便训练误差可以达到0.000000，但是测试数据集上的误差很高。这是典型的过拟合现象。

观察学习的参数。事实上，大部分学到的参数的绝对值比真实参数的绝对值要大一些。


## 使用正则化

下面我们重新初始化模型参数并设置一个正则化参数。

```{.python .input  n=14}
train(5)
```

```{.json .output n=14}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 432x288 with 1 Axes>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "data": {
   "text/plain": "('learned w[:10]:', \n [[ 0.00157753  0.00768491 -0.00250487  0.00650865  0.00434458 -0.00110327\n    0.00045539  0.0072523   0.00239115 -0.00541356]]\n <NDArray 1x10 @cpu(0)>, 'learned b:', \n [0.01004954]\n <NDArray 1 @cpu(0)>)"
  },
  "execution_count": 14,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们发现训练误差虽然有所提高，但测试数据集上的误差有所下降。过拟合现象得到缓解。但打印出的学到的参数依然不是很理想，这主要是因为我们训练数据的样本相对维度来说太少。

## 结论

* 我们可以使用正则化来应对过拟合问题。

## 练习

* 除了正则化、增大训练量、以及使用合适的模型，你觉得还有哪些办法可以应对过拟合现象？
* 如果你了解贝叶斯统计，你觉得$L_2$范数正则化对应贝叶斯统计里的哪个重要概念？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/984)

```{.python .input}
variance 下降了，但是bias比较高，训练需要增加维度
```
