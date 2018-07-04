import torch
from linear import linear_fun
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F


def rec_lin_func(input, hidden, w_hh, w_ih, b_hh, b_ih):
    output = input.matmul(w_ih.t()) + hidden.matmul(w_hh)
    if b_hh is not None:
        output += b_hh
    if b_ih is not None:
        output += b_ih

    return F.tanh(output)


class Rec_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bias=None, activation='tanh'):
        super(Rec_NN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bias = bias

        self.h = torch.Tensor(hidden_size)
        self.cell = Rec_cell(input_size, hidden_size, bias)
         
        self.w_hy = Parameter(torch.Tensor(output_size, hidden_size))

        if self.bias:
            self.b_hy = Parameter(torch.Tensor(output_size))
        else:
            self.b_hy = None
        self.init_params()
        
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise UnsupportedActivationFunction("activation function can only be tanh, sigmoid or relu.")

    def init_params(self):
        torch.nn.init.xavier_uniform(self.w_hy)
        if self.bias:
            self.b_hy.data = torch.randn(self.b_hy.shape)
    
    def forward(self, x, h0):
        self.h = h0
        for i in range(x.size()[1]):
            self.h = self.cell(x[:, i,:], self.h)
        print(self.h.shape)
        print(self.b_hy.shape)
        print(self.w_hy.shape)
        out = linear_fun(self.h, self.w_hy, self.b_hy)
        return self.activation(out)



class Rec_cell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=None):
        super(Rec_cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.w_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.w_hh = Parameter(torch.Tensor(hidden_size, hidden_size))

        if self.bias:
            self.b_hh = Parameter(torch.Tensor(self.hidden_size))
            self.b_ih = Parameter(torch.Tensor(self.hidden_size))
        else:
            self.b_hh = None
            self.b_ih = None
        self.init_params()

    def init_params(self):
        torch.nn.init.xavier_uniform(self.w_ih)
        torch.nn.init.xavier_uniform(self.w_hh)
        if self.bias:
            self.b_ih.data = torch.randn(self.b_ih.size())
            self.b_hh.data = torch.randn(self.b_hh.size())

    def forward(self, input, hidden):
        return rec_lin_func(input, hidden, self.w_hh, self.w_ih, self.b_hh, self.b_ih)


class UnsupportedActivationFunction(Exception):
    """raised when the user pass unsupported activation function."""
    pass