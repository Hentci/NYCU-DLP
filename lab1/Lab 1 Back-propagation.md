# Lab 1 Back-propagation

#### 313551055 柯柏旭

## 1. Introduction



## 2. Experiment setups

#### Hardware overview

```shell
Model Name: MacBook Pro
Model Identifier: Mac15,6
Model Number: MRX33TA/A
Chip: Apple M3 Pro
Total Number of Cores: 11 (5 performance and 6 efficiency)
Memory: 18 GB
```

#### Python version

```sh
Python 3.9.19
```

### A. Sigmoid functions 

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)
```

(refer to lab1 document)

### B. Neural network 

```python
```



### C. Backpropagation 

## 3. Result of testing

### A. Screenshot and comparison figure 

### B. Show the accuracy of your prediction

### C. Learning curve (loss, epoch curve)

### D. Anything you want to present

## 4. Discussion









1. 计算输出层的误差

定义输出层的误差：
 $dZ3 = \frac{\partial L}{\partial Z3}$ 
对于均方误差损失函数，输出层的误差是：
 dZ3 = A3 - y 

2. 计算损失函数对权重的梯度

我们需要计算损失函数 (L) 对权重 (W3) 的梯度 (dW3 = \frac{\partial L}{\partial W3})。

由于 (Z3 = A2 \cdot W3 + b3)，我们可以将 (L) 对 (W3) 的偏导数写成：
 \frac{\partial L}{\partial W3} = \frac{\partial L}{\partial Z3} \cdot \frac{\partial Z3}{\partial W3} 

3. 应用链式法则

首先计算 ( \frac{\partial Z3}{\partial W3} )：
 Z3 = A2 \cdot W3 + b3 
 \frac{\partial Z3}{\partial W3} = A2 

将上面的结果代入链式法则：
 \frac{\partial L}{\partial W3} = \frac{\partial L}{\partial Z3} \cdot A2 

由于 (dZ3 = \frac{\partial L}{\partial Z3})，我们可以得到：
 dW3 = dZ3 \cdot A2 
