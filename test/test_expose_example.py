import unittest
import torch
from networks.OneLinerReLU import ExposeNet
from abZono import Zonotope
from abZono import transform_network


class TestExposeExample(unittest.TestCase):
    
    def test_expose_example(self):
        x = Zonotope(torch.tensor([4.0, 3.0]), torch.tensor([[2.0, 1.0], [1.0, 2.0]]))
        net = ExposeNet()
        transformed_network = transform_network(net)
        print(transformed_network)
        result = transformed_network(x)
        print(result)
        reference = Zonotope(center=torch.tensor([10.0000, 0.1250]), generators=torch.tensor([[4.0000, -0.2500],
                                                                                              [5.0000, 0.2500],
                                                                                              [0.0000, 0.3750]]))
        # Compare to manual result
        self.assertTrue(torch.allclose(result.center, reference.center))
        self.assertTrue(torch.allclose(result.generators, reference.generators))


if __name__ == '__main__':
    unittest.main()
