# TITO Transformer

这个项目的目标是参照 [tinytorch](https://mlsysbook.ai/tinytorch/) 的官方路线，从（硬件以上的）最底层实现开始逐步搭建起足以支撑训练一个 GPT 模型的深度学习训练框架。

由于 tinytorch 是面向机器学习各个领域进行普适化设计的，旨在让学习者“重走、亲手重现”机器学习整个历史；而我们期望设计的 GPT Model 只是其中 NLP 领域的一条分支，因此不是所有内容都在此次学习范围内，主要排除的是在 CV 领域广泛使用的卷积相关算子设计逻辑和实现。

对精通 Pytorch 的同学来说，从表象来看，神经网络的基本单元分两类：结构的基本单元 **Module** (Layer)、数据的基本单元 **Tensor**。

> 一则数据流以 Tensor 形式流入一个由若干 Module 搭建的网络中，与各个 Module 互动产生新的 Tensor，用于输出或更新 Module。

但若深入去看，**Module** 只是一种逻辑上的区分概念，本质上与数据流 Tensor 产生互动的仍然是各种 **Tensor** ：

```python
model = torch.nn.Linear(784, 10)
output = model(x)
# ==== model(x)中进行的 ====
class Linear(nn.Module):
    def __init__(self, ...):
        self.weight = ... # tensor
        self.bias = ... # tensor
    
    def forward(self, x):
        return x @ self.weight + self.bias
```

所以归根结底，张量是神经网络的最基本单元，认识张量、认识张量如何组织（Data Loading、Model Architecture）认识张量如何变化（前向传播、反向传播、优化器）就是整体的学习路线：Tensor -> Layer -> Model，具体到本篇针对 Transformer LLM 的实战中，最终的 Model 就是 GPT。

我们前面还提到，机器学习细分为多个领域，而有些算子或方法是特定于某个领域的。NLP也是如此，包含了将语言数据转换为张量的方法（Tokenizer + Embedding）、将逻辑信息转换为张量的方法（Position Encoding + Attention），这些 LLM 不可或缺的一部分自然也在学习范围内。

学习时长预计 14 天左右，本系列是参与[ Datawhale 静静大佬正在规划中的 mlsys 学习计划](https://github.com/lynnyulinlin-debug/mlsys_on_device) 的产物。

# TITO 教程食用方法
TITO 教程很有意思，旨在推动学习者从渐进的 Notebook 中逐块实现每一个小细节，然后使用对应的脚本进行校验和通过，这里略过安装过程，大概列一下使用方法，从目前的学习计划打卡 issue 中看，也许有人还不太清楚这一点：

```bash
tito module status
```
查看 tito 教程中所有共计 20 个小节的待学习内容，它们在初始情况下都是锁住的，学习者需要像游戏闯关一样，按顺序完成并解锁后续内容。

比如我们即将开始 01_tensor 小节的学习，就需要用：
``` bash
tito module start 01
``` 
先解锁内容，tito 会为你自动打开该节 Notebook，在其中进行学习和填空，待完成认为可以提交了，再在命令行中输入提交命令：
```bash
tito module complete 01
```
tito 会紧接着执行一系列 test 来检查你的实现是否 OK，这时再看 ：
``` bash
tito module status
```
就会发现进度已经变成 1/20，同时 02_activations 小节也随之解锁，后续学习以此类推。
