import argparse
import logging
import numpy as np
import torch
from zonotope import Zonotope
from network_transformer import transform_network
from onnx2torch import convert

parser = argparse.ArgumentParser(
    description='Neural Network Verification using Zonotope relaxation')
parser.add_argument('--net', type=str, metavar='N', help='Path to onnx file', required=True)
parser.add_argument('--center', type=str, metavar='N', help='Path to center file', required=True)
parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
parser.add_argument('--epsilon', type=float, default=0.1, help="epsilon")
parser.add_argument('-d', '--debug', action='store_true', help="Debug mode")

torch.set_grad_enabled(True)
logger = logging.getLogger(__name__)


def main():
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger.debug(args)

    device = 'cpu' #TODO: change to GPU

    net = convert(args.net)
    zono_net = transform_network(net)
    # Load input
    x = np.load(args.center)
    x = torch.from_numpy(x).float()
    epsilon = 2.0 / 255
    true_label = 7

    x = Zonotope.from_l_inf(x, epsilon)
    x.to_device(device)
    y = zono_net(x)

    print(y.center)
    print(y.generators)
    print(y.get_label())


if __name__ == "__main__":
    main()
