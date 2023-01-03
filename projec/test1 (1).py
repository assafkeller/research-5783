import torch
import torch.nn as nn
import torch.nn.functional as F
from parallel_utils1 import ModuleParallel, convParallel
from HyperRes1 import HyperRes,HyperConv,ResBlockMeta
from torch.testing._internal.common_utils import TestCase
from torch.nn.init import calculate_gain
import numpy as np
import random
import unittest
import math
import string
from functools import reduce
from operator import mul



def test_HyperRes_model_init():
    # Test the initialization of HyperRes model
    model = HyperRes(meta_blocks=16, level=[15], device='cpu', bias=True, gray=False, norm_factor=255)
    # Check the value of level attribute
    assert model.level == [0.058823529411764705] #level/norm_factor
    # Check the value of device attribute
    assert model.device == 'cpu'
    # Check the value of inplanes attribute
    assert model.inplanes == 64
    # Check the value of outplanes attribute
    assert model.outplanes == 64
    # Check the value of dilation attribute
    assert model.dilation == 1
    # Check the value of num_parallel attribute
    assert model.num_parallel == 1
    # Check the value of channels attribute
    assert model.channels == 3
    # Calculate gain for ReLU activation
    gain = calculate_gain('relu')
    assert gain == 1.4142135623730951


class TestHyperConv(unittest.TestCase):

    def test_init(self):
        # test that the input parameters are correctly set as instance variables
        levels = [0.5, 1.0, 2.0]
        in_channels = 4
        out_channels = 8
        kernel_size = 3
        stride = 1
        padding = 1
        dilation = 1
        groups = 1
        bias = True
        padding_mode = 'zeros'
        device = 'cpu'
        hyper_conv = HyperConv(levels, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                               padding_mode, device)
        # Check that the input parameters are correctly set as instance variables
        self.assertEqual(hyper_conv.levels, levels)
        self.assertEqual(hyper_conv.in_channels, in_channels)
        self.assertEqual(hyper_conv.out_channels, out_channels)
        self.assertEqual(hyper_conv.kernel_size, kernel_size)
        self.assertEqual(hyper_conv.stride, stride)
        self.assertEqual(hyper_conv.padding, padding)
        self.assertEqual(hyper_conv.dilation, dilation)
        self.assertEqual(hyper_conv.groups, groups)
        self.assertEqual(hyper_conv.bias, bias)
        self.assertEqual(hyper_conv.padding_mode, padding_mode)
        self.assertEqual(hyper_conv.device, device)

    def test_forward(self):
        # test the forward method of the HyperConv class
        levels = [0.5, 1.0, 2.0]
        in_channels = 4
        out_channels = 8
        kernel_size = 3
        stride = 1
        padding = 1
        dilation = 1
        groups = 1
        bias = True
        padding_mode = 'zeros'
        device = 'cpu'
        # Create instance of HyperConv
        hyper_conv = HyperConv(levels, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                               padding_mode, device)

        # create sample input tensors
        x = [torch.randn(2, in_channels, 5, 5) for _ in range(len(levels))]

        # compute output
        out = hyper_conv(x)

        # verify that the output has the correct shape
        for i in range(len(levels)):
            self.assertEqual(out[i].shape, (2, out_channels, 5, 5))

class TestModuleParallel(unittest.TestCase):

    def test_init(self):
        # test that the input module is correctly set as an instance variable
        # Create instance of nn.ReLU
        module = nn.ReLU(inplace=True)
        parallel = ModuleParallel(module)
        # Check that the input module is correctly set as an instance variable
        self.assertEqual(parallel.module, module)

    def test_forward(self):
        # test the forward pass of the ModuleParallel class
        # Create instance of nn.ReLU
        module = nn.ReLU(inplace=True)
        parallel = ModuleParallel(module)

        # create a sample input tensor
        x = [torch.randn(2, 3, 4, 5) for _ in range(3)]

        # compute the output
        out = parallel(x)

        # verify that the output has the correct shape
        for i in range(3):
            self.assertEqual(out[i].shape, (2, 3, 4, 5))

class TestResBlockMeta(unittest.TestCase):

    def test_init(self):
        # test that the input parameters are correctly set as instance variables
        levels = [0.5, 1.0, 2.0]
        ker_n = 4
        inplanes = 8
        device = 'cpu'
        bias = True
        # Create instance of ResBlockMeta
        res_block_meta = ResBlockMeta(levels, ker_n, inplanes, device, bias)
        # Check that the input parameters are correctly set as instance variables
        self.assertEqual(res_block_meta.levels, levels)
        self.assertEqual(res_block_meta.ker_n, ker_n)
        self.assertEqual(res_block_meta.inplanes, inplanes)
        self.assertEqual(res_block_meta.device, device)
        self.assertEqual(res_block_meta.bias, bias)

    def test_forward(self):
        # test the forward pass of the ResBlockMeta class
        levels = [0.5, 1.0, 2.0]
        ker_n = 4
        inplanes = 4
        device = 'cpu'
        bias = True
        # Create instance of ResBlockMeta
        res_block_meta = ResBlockMeta(levels, ker_n, inplanes, device, bias)

        # create sample input tensors
        x = [torch.randn(2, ker_n, 5, 5) for _ in range(len(levels))]

        # compute the output
        out = res_block_meta(x)

        # verify that the output has the correct shape
        for i in range(len(levels)):
            self.assertEqual(out[i].shape, (2, inplanes, 5, 5))


if __name__ == '__main__':
    unittest.main()
