# Needle
Completed an online course on 10-714 [Deep Learning Systems](https://dlsyscourse.org/) offered by CMU in order to delve into the internals of PyTorch and TensorFlow, and understand how they function at a fundamental level.

Designed and built a deep learning library called **Needle**, comparable to a very minimal version of PyTorch or TensorFlow, capable of efficient GPU-based operations, automatic differentiation of all implemented functions, and the necessary modules to support parameterized layers, loss functions, data loaders, and optimizers.

![](https://github.com/0324wy/needle/blob/main/layersInDLF.png)

## Project 0

Build a basic softmax regression algorithm, plus a simple two-layer neural network. Create these implementations both in native Python (using the numpy library), and (for softmax regression) in native C/C++. 

&#x2705; A basic `add` function

&#x2705; Loading MNIST data:  `parse_mnist` function

&#x2705;Softmax loss: `softmax_loss` function

&#x2705;Stochastic gradient descent for softmax regression

&#x2705;SGD for a two-layer neural network

&#x2705;Softmax regression in C++

## Project 1

Build a basic **automatic differentiation** framework, then use this to re-implement the simple two-layer neural network we used for the MNIST digit classification problem in HW0.

&#x2705;Implementing forward computation

- &#x2705;PowerScalar
- &#x2705;EWiseDiv
- &#x2705;DivScalar
- &#x2705;MatMul
- &#x2705;Summation
- &#x2705;BroadcastTo
- &#x2705;Reshape
- &#x2705;Negate
- &#x2705;Transpose

&#x2705;Implementing backward computation

- &#x2705;EWiseDiv
- &#x2705;DivScalar
- &#x2705;MatMul
- &#x2705;Summation
- &#x2705;BroadcastTo
- &#x2705;Reshape
- &#x2705;Negate
- &#x2705;Transpose

&#x2705;Topological sort: allow us to traverse through (forward or backward) the computation graph, computing gradients along the way

&#x2705;Implementing reverse mode differentiation

&#x2705;Softmax loss

&#x2705;SGD for a two-layer neural network

## Project 2

Implement a **neural network library** in the needle framework.

&#x2705;Implement a few different methods for weight initialization

&#x2705;Implement additional modules

- &#x2705;Linear: `needle.nn.Linear` class
- &#x2705;ReLU:`needle.nn.ReLU` class
- &#x2705;Sequential: `needle.nn.Sequential` class
- &#x2705;LogSumExp: `needle.ops.LogSumExp` class
- &#x2705;SoftmaxLoss: `needle.nn.SoftmaxLoss` class
- &#x2705;LayerNorm1d: `needle.nn.LayerNorm1d` class
- &#x2705;Flatten: `needle.nn.Flatten` class
- &#x2705;BatchNorm1d: `needle.nn.BatchNorm1d` class
- &#x2705;Dropout: `needle.nn.Dropout` class
- &#x2705;Residual: `needle.nn.Residual` class

&#x2705;Implement the `step` function of the following optimizers.

- &#x2705;SGD: `needle.optim.SGD` class
- &#x2705;Adam: `needle.optim.Adam` class

&#x2705;Implement two data primitives: `needle.data.DataLoader` and `needle.data.Dataset`

- &#x2705;Transformations: `RandomFlipHorizontal` function and `RandomFlipHorizontal` class
- &#x2705;Dataset: `needle.data.MNISTDataset` class
- &#x2705;Dataloader: `needle.data.Dataloader` class

&#x2705;Build and train an MLP ResNet
- &#x2705;ResidualBlock: `ResidualBlock` function
- &#x2705;MLPResNet: `MLPResNet` function
- &#x2705;Epoch: `epoch` function
- &#x2705;Train Mnist: `train_mnist` function

## Project 3: Building an NDArray library


Build a simple backing library for the processing that underlies most deep learning systems: **the n-dimensional array** (a.k.a. the NDArray). 

&#x2705;Python array operations
- &#x2705;reshape: `reshape` function
- &#x2705;permute: `permute` function


