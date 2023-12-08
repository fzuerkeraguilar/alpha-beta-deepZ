import unittest
import torch
from networks.OneLinerReLU import ExposeNet
from abZono import Zonotope
from abZono import trans_layer, trans_network


class TestExposeExample(unittest.TestCase):
    def test_expose_example(self):
        x = Zonotope(torch.tensor([4.0, 3.0]), torch.tensor([[2.0, 1.0], [1.0, 2.0]]))
        net = ExposeNet()
        trans_layers = [trans_layer(layer) for layer in net.layers]
        result = trans_network([trans_layers[0]])(x)
        print(result)
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()
