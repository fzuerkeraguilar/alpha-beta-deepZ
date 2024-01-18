import unittest
from abZono.network_transformer import transform_network
from abZono.zonotope import Zonotope
from abZono.example_vnnlib import get_num_inputs_outputs, read_vnnlib_simple
from abZono.utils import numpy_dtype_to_pytorch_dtype
from onnx2torch import convert
import torch

NETWORK_PATH = './vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx'
SPEC_PATH = './vnncomp2022_benchmarks/benchmarks/mnist_fc/vnnlib/prop_0_0.03.vnnlib'


class TestLossReduction(unittest.TestCase):

    def test_loss_reduction(self):
        # Load the network and input zonotope
        network = convert(NETWORK_PATH)

        input_size, input_shape, output_size, output_shape, dtype = get_num_inputs_outputs(NETWORK_PATH)
        dtype = numpy_dtype_to_pytorch_dtype(dtype)
        spec = read_vnnlib_simple(SPEC_PATH, input_size, output_size)
        x = Zonotope.from_vnnlib(spec[0][0], input_shape, dtype)

        # Transform the network and propagate the zonotope
        transform_network(network, optimize_alpha=True)
        y = network(x)
        initial_loss = y.vnnlib_loss(spec[0][1][0])
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        
        loss = None  # Assign a default value to the 'loss' variable
        
        for _ in range(10):
            optimizer.zero_grad()
            y = network(x)
            loss = y.vnnlib_loss(spec[0][1][0])
            loss.backward()
            optimizer.step()
            print(loss)

        self.assertTrue(initial_loss > loss)

