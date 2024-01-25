import unittest
from abZono.__main__ import load_net_and_input_zonotope
import torch
import os

NETWORK_PATH = './test/vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx' \
    if os.path.exists('./test/vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx') \
    else './vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx'
SPEC_PATH = './test/vnncomp2022_benchmarks/benchmarks/mnist_fc/vnnlib/prop_0_0.03.vnnlib' \
    if os.path.exists('./test/vnncomp2022_benchmarks/benchmarks/mnist_fc/vnnlib/prop_0_0.03.vnnlib') \
    else './vnncomp2022_benchmarks/benchmarks/mnist_fc/vnnlib/prop_0_0.03.vnnlib'


class TestLossReduction(unittest.TestCase):

    def test_loss_reduction(self):
        # Load the network and input zonotope
        network, x, spec = load_net_and_input_zonotope(NETWORK_PATH, SPEC_PATH, 'cpu')
        y = network(x)
        initial_loss = y.vnnlib_loss(spec)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        
        loss = 0  # Assign a default value to the 'loss' variable
        
        for _ in range(10):
            optimizer.zero_grad()
            y = network(x)
            loss = y.vnnlib_loss(spec)
            loss.backward()
            optimizer.step()
            print(loss)

        self.assertTrue(initial_loss > loss)

