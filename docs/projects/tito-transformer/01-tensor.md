# 01 - 张量 网络的基本单元

数组是**数据结构课程**中最为核心的概念，它可以作为后续各类复杂数据类型的构建基石，对链表、树、堆、图等的概念理解都离不开对数组的认识。

同理，作为**机器学习领域**中最为核心的概念，Tensor 也值得你由浅入深地细致研究。在这一节中，我们会从概念分析开始，逐步亲手实现一个类似 `torch.Tensor` 的 `class Tensor`，也将是我们后续内容构建自己版本 `Module/Layer/Model` 时使用的依赖。

在本节的学习中，请带着如下问题：

1. Tensor 需要是什么样的？
2. Tensor 需要能做什么？
3. 为什么要考虑**优化**？

## Make a Tensor

在 Project page 中已经提到，Tensor 是神经网络中流动的基本单元，也是计算的基本单元，因此它必须有能够参与运算的数学性质，即**代数性（Arithmetic）**；同时，Tensor 可以被认为是矩阵概念的延伸，自然在高维场景下继承了矩阵所具有的**基本属性（Properties）** 和计算规律。综合这些，可以给 Tensor 这样定义：**一个支持在不同维度下存储数据的容器**。
为了方便起见，在 `numpy.array` 的基础上建立 Tensor。

1. 基本属性：一个 Tensor 自身具有的静态或动态属性，与场景或环境无关。
    - 形状/尺寸：shape, size（继承自矩阵）
    - 包含数据格式：dtype
    - 包含数据内容：data
2. 运算属性：
    - 代数运算：加减乘除代数运算规则。
    - 矩阵运算：继承矩阵乘法规则
3. 操作变换：
    - 形状尺寸的变换：reshape, transpose
    - 内存布局的变换：contiguous, copy
    - 数据内容的变换：sum, mean, max

那么现在，我们抽象出的 Tensor 类大概长这样：

```python
class Tensor:

    # 张量的基本属性
    def __init__(self, ):
        self.shape = 
        self.size = 
        self.data = # 这会是一个 numpy.array
        self.dtype = 
    @property
    def ndim(self):
        return len(self.shape)
    def numel(self):
        return self.size
    
    # 控制 Tensor 对象格式化输出
    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape})"
    def __str__(self):
        return f"Tensor(data={self.data})"
    def numpy(self):
        # 返回内部的 numpy array
        return self.data
    
    # 张量的运算属性
    def __add__(self, other): # +
    def __radd__(self, other):
    def __sub__(self, other): # -
    def __rsub__(self, other):
    def __mul__(self, other): # *
    def __rmul__(self, other):
    def __truediv__(self, other): # /
    def __rtruediv__(self, other):
    def __matmul__(self.other): # @

    # 张量的操作变换
    def reshape(self, shape):
    def transpose(self, dim1, dim2):
    def sum(self, axis, keepdims):
    ...
```

但是，现有的抽象定义只考虑了 Tensor 作为一个基础运算对象的各类性质，不要忘了神经网络中非常重要的概念——反向传播。反向传播利用各个元素项的偏导数更新权重参数，而在网络中，权重参数和中间激活都是张量，因此张量还应当具有记录自身梯度和维护反向传播的能力，这些也应属于其基本属性的一部分。需要增加：

```python
def __init__(...):
    ...
    self.requires_grad = True / False
    self.grad = 
    self._grad_fn = 
    self._graph_released = 

def backward():
def zero_grad():
```

整体框架搭好了，先来看最初的步骤——初始化该如何实现。踩在 numpy 的肩膀上，只需要把任意有效的数据转换成 `np.array` 包装到 `Tensor.data` ，其他属性也可以继承自 array 对象，再补充其不具有的梯度相关参数：

```python
def __init__(self, data, requires_grad=False):
    self.data = np.array(data, dtype=np.float32)
    self.shape = self.data.shape
    self.size = self.data.size
    self.dtype = self.data.dtype

    self.requires_grad = requires_grad
    self.grad = None
    self._grad_fn = None
    self._graph_released = False
```
然而在这种实现下，当尝试把多个 Tensor 组成的序列组装成一个新的 Tensor （在深度学习中，难免会遇到这种情况，比如组合多个 Transformer Block 的中间输出为 List），`np.array`就会尝试转换 Tensor 到 data 类型，而不是 Tensor 中包含的数据本身，从而出现错误。增加判断，让它能够读取到真正的 data：

```python
if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], Tensor):
    data = np.stack([t.data for t in data])
self.data = np.array([data, dtype=np.float32])
...
```

现在我们的 Tensor 类已经可以顺利创建对象了。承载数据只是第一步，接下来我们将了解如何**操作一个 Tensor** 。

## Operation vs Tensor

如何操作一个 Tensor ？这不是一个无聊的问题，在设计**系统**时，不能只考虑实现难易，还应尽可能考虑兼容性、可用性。复制张量时是保持内容地址仅复制索引，还是连带数据一同单独拷贝？修改张量时是原地修改，还是创建新的副本……这些问题的选择不仅影响性能，更关乎系统运行的稳定性和正确性。

此外，回顾在创建 Tensor 类时提到的一点：神经网络中需要反向传播。而反向传播依赖于累计梯度和正向中间值，因此 `backward` 不仅与旧的 `backward` 相关，还和当前的 `forward` 相关。在运算过程中的任何一项操作都会对网络中计算结构、梯度网络产生影响。

这时就必须考虑一个问题：**关于 Tensor 的 Operation 是否要和 Tensor 绑定？**

如果你接触过面向对象编程，会有直觉认为与类实例相关的操作都应帮定为类的方法，以体现其在逻辑上的整体性：`class Cat -> func Meow`。那么，假如我们把有关 Tensor 的操作也写成内部方法，会发生什么？以矩阵乘法`mul`为例，考虑以下两点：

1. 我们此前定义的张量本身只记录自己的数值和梯度（如果有），它不知道自己在网络中的位置，也不知道自己经过了什么计算。
2. 不同的函数会产生不同的反向传播逻辑，即导数计算公式是不同的。

进行前向传播非常简单，调用 `X.mul(other)`:

```python
def mul(self, other):
    out = Tensor(self.data * other.data)
    return out
```
同时要为反向传播做准备，实例必须保存原始输入和中间值，并根据运算类型使用相应的反向过程：
```python
def mul(self, other):
    out = Tensor(self.data * other.data)
    if self.requires_grad or other.requires_grad:
        out.requires_grad = True
        # 在这个 Tensor 内部存下反向所需的内容
        out._inputs = (self, other)
        out._ctx = (self.data, other.data)
        out._backward = self.mul_backward # 指定特定反向计算
    return out

def mul_backward(self, ctx):
    # 从 Tensor 保存的内容中进行反传计算
```

发现在要考虑反向传播的情况下，Tensor 就不只是拥有前一部分讨论的基础属性了，而多了许多额外属性：`_inputs` 记录它在产生时的数据源，`_ctx` 记录它的数据源的内容，`_backward` 要记录它参与的运算类型，指定反向时要用的反传函数。现在 Tensor 类的定义就不再完全符合我们原本定义的 “基本数值性质”，变得臃肿杂乱。

> 值得一提的是，self.some_method 是绑定方法，它内部持有 self 的引用。把它赋给 out._backward，就等于让 out 间接持有了整个 self——可能形成引用循环，意外地长期扣住整个 Tensor 及其数据。

影响实现方案的思考是：**梯度究竟是数据相关，还是运算相关**？一个张量可能参与无数种运算，而一种运算有其确定的导数推导式；梯度由运算规则决定，但是又作用于数据，所以任何一方都不能单独包揽对梯度的责任。在这一观点的引导下，我们要尽量将复杂性降低，减少多重耦合，因此将运算和张量解耦，梯度运算不直接与张量相连，而是通过运算间接利用张量。

![20260917154912](https://dodoaix-blog-1307847568.cos.ap-guangzhou.myqcloud.com/blog/tito/20260917154912.png)

因此，让 Tensor 类保持此前的干净结构，把各类操作统一抽象为单独的 Function 类，每一类操作单独定义自己的前后向传播运算规则，通过与操作绑定的 `_ctx` 记录单次操作反向所需的上下文信息。并且这种设计还带来了额外的好处：当需要新增自定义操作/函数时，只需要继承 Function 类并实现前后向逻辑就能加入到网络中，不必改动 Tensor 类。

> _ctx 保存的正是反向传播所依赖的正向中间值。

## Make a Operation


## Operate a Tensor