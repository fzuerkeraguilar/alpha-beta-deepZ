import unittest
from abZono.network_transformer import transform_network
from abZono.zonotope import Zonotope
from abZono.example_vnnlib import get_num_inputs_outputs, read_vnnlib_simple
from abZono.utils import numpy_dtype_to_pytorch_dtype
from abZono.network_transformer import transform_network_fx
from abZono.__main__ import load_net_and_input_zonotope
from onnx2torch import convert
import torch
import copy
import os

NETWORK_PATH = './test/vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx' \
    if os.path.exists('./test/vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx') \
    else './vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx'
SPEC_PATH = './test/vnncomp2022_benchmarks/benchmarks/mnist_fc/vnnlib/prop_0_0.03.vnnlib' \
    if os.path.exists('./test/vnncomp2022_benchmarks/benchmarks/mnist_fc/vnnlib/prop_0_0.03.vnnlib') \
    else './vnncomp2022_benchmarks/benchmarks/mnist_fc/vnnlib/prop_0_0.03.vnnlib'


class TestZonotopePropagation(unittest.TestCase):

    def test_center_propagation(self):
        # Load the network and input zonotope
        network = convert(NETWORK_PATH)

        input_size, input_shape, output_size, output_shape, dtype = get_num_inputs_outputs(NETWORK_PATH)
        dtype = numpy_dtype_to_pytorch_dtype(dtype)
        spec = read_vnnlib_simple(SPEC_PATH, input_size, output_size)
        input_zonotope = Zonotope.from_vnnlib(spec[0][0], input_shape, dtype)

        # Propagate the center of the input zonotope through the network
        center_through_network = network(input_zonotope.center)
        # Transform the network and propagate the zonotope
        network_copy = copy.deepcopy(network)
        output_zonotope = transform_network(network_copy)(input_zonotope)

        # Check if the center of the output zonotope is contained in the propagated zonotope
        self.assertTrue(output_zonotope.contains_point(center_through_network))

    def test_random_points_in_output_zonotope(self):
        # Load the network and input zonotope
        network = convert(NETWORK_PATH)

        input_size, input_shape, output_size, output_shape, dtype = get_num_inputs_outputs(NETWORK_PATH)
        dtype = numpy_dtype_to_pytorch_dtype(dtype)
        spec = read_vnnlib_simple(SPEC_PATH, input_size, output_size)
        input_zonotope = Zonotope.from_vnnlib(spec[0][0], torch.Size(input_shape), dtype)

        random_points = [input_zonotope.random_point() for _ in range(10000)]

        for point in random_points:
            self.assertTrue(input_zonotope.contains_point(point))

        network_copy = copy.deepcopy(network)

        # Transform the network
        output_zonotope = transform_network_fx(network_copy, input_zonotope.center)(input_zonotope)

        # Select 10 random points from the input zonotope
        points_through_original_network = [network(point) for point in random_points]

        # Check if each transformed point is in the output zonotope
        for point in points_through_original_network:
            self.assertTrue(output_zonotope.contains_point(point))

    def test_random_points_in_output_zonotope_with_alpha(self):
        # Load the network and input zonotope
        original_network = convert(NETWORK_PATH)
        network, x, _ = load_net_and_input_zonotope(NETWORK_PATH, SPEC_PATH, 'cpu')

        random_points = [x.random_point() for _ in range(10000)]

        for point in random_points:
            self.assertTrue(x.contains_point(point))

        # Transform the network
        y = network(x)

        # Select random points from the input zonotope
        points_through_original_network = [original_network(point) for point in random_points]

        # Check if each transformed point is in the output zonotope
        for point in points_through_original_network:
            self.assertTrue(y.contains_point(point))

    def test_fuzzy_propagation_after_optimization(self):
        original_network = convert(NETWORK_PATH)
        network, x, spec = load_net_and_input_zonotope(NETWORK_PATH, SPEC_PATH, 'cpu')

        random_points = [x.random_point() for _ in range(10000)]
        for point in random_points:
            self.assertTrue(x.contains_point(point))

        points_through_original_network = [original_network(point) for point in random_points]

        y = network(x)

        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        for _ in range(100):
            optimizer.zero_grad()
            y = network(x)
            loss = y.vnnlib_loss(spec)
            loss.backward()
            optimizer.step()

        for point in points_through_original_network:
            self.assertTrue(y.contains_point(point))


if __name__ == '__main__':
    unittest.main()
