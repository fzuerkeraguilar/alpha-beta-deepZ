import argparse
import logging
from example_vnnlib import get_num_inputs_outputs, read_vnnlib_simple
import numpy as np
import torch
from zonotope import Zonotope
from network_transformer import transform_network
from onnx2torch import convert
from utils import numpy_dtype_to_pytorch_dtype

parser = argparse.ArgumentParser(
    description='Neural Network Verification using Zonotope relaxation')
parser.add_argument('--net', type=str, metavar='N',
                    help='Path to onnx file', required=True)
parser.add_argument('--spec', type=str, metavar='N',
                    help='Path to vnnlib file')
parser.add_argument('--center', type=str, metavar='N',
                    help='Path to center file')
parser.add_argument('--epsilon', type=float, default=0.1, help="epsilon")
parser.add_argument('--true-label', type=int, help="True label")
parser.add_argument('--cpu', action='store_true',
                    help='Use CPU instead of GPU')
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

    if not args.spec:
        if not (args.center and args.epsilon and args.true_label):
            raise Exception(
                "Please provide either spec or center, epsilon and true label")
        else:
            center = np.load(args.center)
            center = torch.from_numpy(center).float()
            epsilon = float(args.epsilon)
            true_label = int(args.true_label)
            x = Zonotope.from_l_inf(center, epsilon)
    else:
        input_size, input_shape, output_size, output_shape, dtype = get_num_inputs_outputs(
            args.net)
        logger.debug("Input size: {}, Output size: {}".format(
            input_size, output_size))
        logger.debug("Input shape: {}, Output shape: {}".format(
            input_shape, output_shape))
        dtype = numpy_dtype_to_pytorch_dtype(dtype)
        logger.debug("Dtype: {}".format(dtype))
        spec = read_vnnlib_simple(args.spec, input_size, output_size)
        x = Zonotope.from_vnnlib(spec[0][0], input_shape, dtype)

    device = 'cpu'  # TODO: change to GPU

    net = convert(args.net)
    logger.debug(net)
    zono_net = transform_network(net, optimize_alpha=True)
    logger.debug(zono_net)
    y = zono_net(x)
    optimizer = torch.optim.Adam(zono_net.parameters(), lr=0.01)

    # Optimization loop
    for i in range(100):
        y = zono_net(x)
        optimizer.zero_grad()
        loss = y.vnnlib_loss(spec[0][1][0])
        loss.backward(retain_graph=True)
        optimizer.step()
        print(loss.item())




if __name__ == "__main__":
    main()
