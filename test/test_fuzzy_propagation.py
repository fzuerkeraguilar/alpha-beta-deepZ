import unittest
from abZono.network_transformer import transform_network
from abZono.zonotope import Zonotope
from abZono.example_vnnlib import get_num_inputs_outputs, read_vnnlib_simple
from abZono.utils import numpy_dtype_to_pytorch_dtype
from onnx2torch import convert
import copy

NETWORK_PATH = './test/vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx'
SPEC_PATH = './test/vnncomp2022_benchmarks/benchmarks/mnist_fc/vnnlib/prop_0_0.03.vnnlib'


class TestZonotopePropagation(unittest.TestCase):


    def test_random_points_in_output_zonotope(self):
        # Load the network and input zonotope
        network = convert(NETWORK_PATH)

        input_size, input_shape, output_size, output_shape, dtype = get_num_inputs_outputs(NETWORK_PATH)
        dtype = numpy_dtype_to_pytorch_dtype(dtype)
        spec = read_vnnlib_simple(SPEC_PATH, input_size, output_size)
        input_zonotope = Zonotope.from_vnnlib(spec[0][0], input_shape, dtype)

        ten_random_points = [input_zonotope.random_point() for _ in range(10)]

        for point in ten_random_points:
            self.assertTrue(input_zonotope.contains_point(point))

        network_copy = copy.deepcopy(network)

        # Transform the network
        output_zonotope = transform_network(network_copy)(input_zonotope)

        # Select 10 random points from the input zonotope
        ten_random_points_through_network = [network(point) for point in ten_random_points]

        # Check if each transformed point is in the output zonotope
        for point in ten_random_points_through_network:
            self.assertTrue(output_zonotope.contains_point(point))

    def test_random_points_in_output_zonotope_with_alpha(self):
        # Load the network and input zonotope
        network = convert(NETWORK_PATH)

        input_size, input_shape, output_size, output_shape, dtype = get_num_inputs_outputs(NETWORK_PATH)
        dtype = numpy_dtype_to_pytorch_dtype(dtype)
        spec = read_vnnlib_simple(SPEC_PATH, input_size, output_size)
        input_zonotope = Zonotope.from_vnnlib(spec[0][0], input_shape, dtype)

        ten_random_points = [input_zonotope.random_point() for _ in range(10)]

        for point in ten_random_points:
            self.assertTrue(input_zonotope.contains_point(point))

        network_copy = copy.deepcopy(network)

        # Transform the network
        output_zonotope = transform_network(network_copy, optimize_alpha=True)(input_zonotope)

        # Select 10 random points from the input zonotope
        ten_random_points_through_network = [network(point) for point in ten_random_points]

        # Check if each transformed point is in the output zonotope
        for point in ten_random_points_through_network:
            self.assertTrue(output_zonotope.contains_point(point))

if __name__ == '__main__':
    unittest.main()