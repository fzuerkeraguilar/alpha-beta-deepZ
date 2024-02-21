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

NETWORK_PATH = './test/vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x4.onnx' \
    if os.path.exists('./test/vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x4.onnx') \
    else './vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x4.onnx'
SPEC_PATH = './test/vnncomp2022_benchmarks/benchmarks/mnist_fc/vnnlib/prop_6_0.03.vnnlib' \
    if os.path.exists('./test/vnncomp2022_benchmarks/benchmarks/mnist_fc/vnnlib/prop_6_0.03.vnnlib') \
    else './vnncomp2022_benchmarks/benchmarks/mnist_fc/vnnlib/prop_6_0.03.vnnlib'


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

        random_points = [input_zonotope.center] + [input_zonotope.random_point() for _ in range(10)]

        for point in random_points:
            self.assertTrue(input_zonotope.contains_point(point))

        network_copy = copy.deepcopy(network)

        # Transform the network
        y = transform_network_fx(network_copy, input_zonotope.center)(input_zonotope)

        # Select 10 random points from the input zonotope
        points_through_original_network = [network(point) for point in random_points]

        correct_points = 0
        # Check if each transformed point is in the output zonotope
        for point in points_through_original_network:
            if y.contains_point(point):
                correct_points += 1

        if correct_points != len(points_through_original_network):
            print(f'Points correctly: {correct_points}/{len(points_through_original_network)}, {correct_points}')
        self.assertEqual(correct_points, len(points_through_original_network))

    def test_random_points_in_output_zonotope_with_alpha(self):
        # Load the network and input zonotope
        original_network = convert(NETWORK_PATH)
        network, x, _ = load_net_and_input_zonotope(NETWORK_PATH, SPEC_PATH, 'cpu')

        random_points = [x.center] + [x.random_point() for _ in range(100)]

        for point in random_points:
            self.assertTrue(x.contains_point(point))

        y = network(x)

        points_through_original_network = [original_network(point) for point in random_points]

        for i, point in enumerate(points_through_original_network):
            self.assertTrue(y.contains_point(point))

    def test_fuzzy_propagation_after_optimization(self):
        original_network = convert(NETWORK_PATH)
        network, x, spec = load_net_and_input_zonotope(NETWORK_PATH, SPEC_PATH, 'cpu')

        random_points = [x.center] + [x.random_point() for _ in range(100)]
        for point in random_points:
            self.assertTrue(x.contains_point(point))

        points_through_original_network = [original_network(point) for point in random_points]

        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        for i in range(100):
            optimizer.zero_grad()
            y = network(x)
            loss = y.vnnlib_loss(spec)
            loss.backward()
            optimizer.step()

            print(f'Loss: {loss.item()}')
            print(f'Iteration: {i}')
            correct_points = 0
            wrong_points = []
            # Check if each transformed point is in the output zonotope
            for idx, point in enumerate(points_through_original_network):
                if y.contains_point(point):
                    correct_points += 1
                else:
                    wrong_points.append((idx, point))

            if correct_points != len(points_through_original_network):
                print(f"Failed at iteration {i}")
                print(f'Correct points: {correct_points}/{len(points_through_original_network)} of {correct_points}')
            self.assertEqual(correct_points, len(points_through_original_network))

    def test_intermediate_propagation(self):
        original_network = convert(NETWORK_PATH)
        zono_network, x, spec = load_net_and_input_zonotope(NETWORK_PATH, SPEC_PATH, 'cpu')

        random_points = [x.center] + [x.random_point() for _ in range(10)]
        for i, point in enumerate(random_points):  # i for debugging
            self.assertTrue(x.contains_point(point))

        flatten_points = [original_network.Flatten(point) for point in random_points]
        flatten_zono = zono_network.Flatten_zono(x)
        self.assertTrue(torch.all(flatten_zono.center == flatten_points[0]))
        for i, point in enumerate(flatten_points):
            self.assertTrue(flatten_zono.contains_point(point))

        gemm_points = [original_network.Gemm(point) for point in flatten_points]
        gemm_zono = zono_network.Gemm_zono(flatten_zono)
        self.assertTrue(torch.all(gemm_zono.center == gemm_points[0]))
        for i, point in enumerate(gemm_points):
            self.assertTrue(gemm_zono.contains_point(point))

        relu_points = [original_network.Relu(point) for point in gemm_points]
        relu_zono = zono_network.Relu_zono(gemm_zono)
        self.assertTrue(relu_zono.contains_point_box(relu_points[0]))
        for i, point in enumerate(relu_points):
            self.assertTrue(relu_zono.contains_point(point))

        gemm_1_points = [original_network.Gemm_1(point) for point in relu_points]
        gemm_1_zono = zono_network.Gemm_1_zono(relu_zono)
        self.assertTrue(gemm_1_zono.contains_point_box(gemm_1_points[0]))
        for i, point in enumerate(gemm_1_points):
            self.assertTrue(gemm_1_zono.contains_point(point))

        relu_1_points = [original_network.Relu_1(point) for point in gemm_1_points]
        relu_1_zono = zono_network.Relu_1_zono(gemm_1_zono)
        for i, point in enumerate(relu_1_points):
            self.assertTrue(relu_1_zono.contains_point(point))

        gemm_2_points = [original_network.Gemm_2(point) for point in relu_1_points]
        gemm_2_zono = zono_network.Gemm_2_zono(relu_1_zono)
        for i, point in enumerate(gemm_2_points):
            self.assertTrue(gemm_2_zono.contains_point(point))


if __name__ == '__main__':
    unittest.main()
