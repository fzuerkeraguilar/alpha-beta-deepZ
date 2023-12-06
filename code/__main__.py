import argparse
import logging
import torch
from zonotope import Zonotope

parser = argparse.ArgumentParser(description='Neural Network Verification using Zonotope relaxation')
parser.add_argument('--net', type=str, default='mnist', metavar='N',
                    help='network to use: mnist, cifar')
parser.add_argument('--spec', type=str, required=True, help="specification file")
parser.add_argument('--cpu', action='store_true', help='use CPU instead of GPU')
parser.add_argument('--epsilon', type=float, default=0.1, help="epsilon")
parser.add_argument('-d', '--debug', action='store_true', help="debug mode")

torch.set_grad_enabled(True)
logger = logging.getLogger(__name__)

def main():
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger.debug(args)
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")

    with open(args.spec) as f:
        lines = f.readlines()
        true_label = int(lines[0])
        input_size = tuple([int(x) for x in lines[1].split()])
        input_values = [float(x) for x in lines[2].split()]
        input_values = torch.tensor(input_values).reshape(input_size)
        input_zono = Zonotope(input_values, torch.ones_like(input_values) * args.epsilon)


if __name__ == "__main__":
    main()