"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        s_weight = init.kaiming_uniform(in_features, out_features)
        s_bias = None
        if bias is True:
            s_bias = ops.reshape(init.kaiming_uniform(out_features, None), (1, out_features))
        self.weight = Parameter(s_weight)
        self.bias = Parameter(s_bias)

    def forward(self, X: Tensor) -> Tensor:
        result = ops.matmul(X, self.weight)
        if self.bias is not None:
            return result + ops.broadcast_to(self.bias, result.shape)
        return result



class Flatten(Module):
    def forward(self, X):
        return ops.reshape(X, (X.shape[0], -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        z_exp = ops.exp(logits)
        z_exp_sum = ops.summation(z_exp, axes=1)
        z_log = ops.log(z_exp_sum)
        y_one_hot = Tensor(init.one_hot(logits.shape[1], y))
        z_sum = ops.summation(logits * y_one_hot, axes=1)
        return ops.summation(z_log - z_sum) / y_one_hot.shape[0]



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(self.dim))
        self.bias = Parameter(init.zeros(self.dim))
        self.running_mean = init.zeros(self.dim)
        self.running_var = init.ones(self.dim)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        features = x.shape[1]

        sums = ops.summation(x, axes=0)
        mean = ops.divide_scalar(sums, batch_size)
        tmp = ops.reshape(mean, (1, -1))
        broadcast_mean = ops.broadcast_to(tmp, x.shape)

        sub = x - broadcast_mean
        sub2 = ops.power_scalar(sub, 2)
        var = ops.summation(ops.divide_scalar(sub2, batch_size), axes=0)
        broadcast_var = ops.broadcast_to(ops.reshape(var, (1, -1)), x.shape)
        nominator = ops.power_scalar(broadcast_var + self.eps, 0.5)

        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape)
        out = broadcast_weight * (x - broadcast_mean) / nominator + broadcast_bias

        if self.training is True:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            broadcast_mean = ops.broadcast_to(ops.reshape(self.running_mean, (1, -1)), x.shape)
            broadcast_var = ops.broadcast_to(ops.reshape(self.running_var, (1, -1)), x.shape)
            nominator = ops.power_scalar(broadcast_var + self.eps, 0.5)
            out = broadcast_weight * (x - broadcast_mean) / nominator + broadcast_bias
        return out

class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(self.dim))
        self.bias = Parameter(init.zeros(self.dim))

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        features = x.shape[1]

        sums = ops.summation(x,axes=1)
        mean = ops.divide_scalar(sums, features)
        tmp = ops.reshape(mean, (-1, 1))
        broadcast_mean = ops.broadcast_to(tmp, x.shape)

        sub = x - broadcast_mean
        sub2 = ops.power_scalar(sub, 2)
        var = ops.summation(ops.divide_scalar(sub2, features), axes=1)
        broadcast_var = ops.broadcast_to(ops.reshape(var, (-1, 1)), x.shape)

        nominator = ops.power_scalar(broadcast_var + self.eps, 0.5)

        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape)
        out = broadcast_weight * (x - broadcast_mean) / nominator + broadcast_bias
        return out


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training is True:
            mask = init.randb(*x.shape, p = 1 - self.p) / (1 - self.p)
            return x * mask


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)



