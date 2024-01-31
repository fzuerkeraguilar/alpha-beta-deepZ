import argparse
import logging
from .example_vnnlib import get_num_inputs_outputs, read_vnnlib_simple
import numpy as np
import torch
from abZono.zonotope import Zonotope
from abZono.network_transformer import transform_network_fx, transform_network
from onnx2torch import convert
from .utils import numpy_dtype_to_pytorch_dtype
import csv
import os
import time

parser = argparse.ArgumentParser(
    description='Neural Network Verification using Zonotope relaxation')
parser.add_argument('--net', type=str, metavar='N',
                    help='Path to onnx file')
parser.add_argument('--spec', type=str, metavar='N',
                    help='Path to vnnlib file')
parser.add_argument('--center', type=str, metavar='N',
                    help='Path to center file')
parser.add_argument('--epsilon', type=float, default=0.1, help="epsilon")
parser.add_argument('--true-label', type=int, help="True label")
parser.add_argument('--csv', type=str, help="instances.csv file")
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

    device = torch.device('cpu') if args.cpu or not torch.cuda.is_available() else torch.device('cuda')
    logger.debug("Using device: {}".format(device))

    instances = []

    if not args.spec and not args.csv:
        if not (args.center and args.epsilon and args.true_label):
            raise Exception(
                "Please provide either spec or center, epsilon and true label")
        else:
            center = np.load(args.center)
            center = torch.from_numpy(center).float()
            epsilon = float(args.epsilon)
            true_label = int(args.true_label)
            x = Zonotope.from_l_inf(center, epsilon)
            net = convert(args.net)
    elif args.spec:
        net, x, output_spec = load_net_and_input_zonotope(args.net, args.spec, device)
        instances.append((args.net, args.spec, net, x, output_spec))
    elif args.csv:
        with open(args.csv, 'r') as f:
            # Get dir of csv file
            csv_dir = os.path.dirname(args.csv)

            reader = csv.reader(f)
            for row in reader:
                model_path = os.path.join(csv_dir, row[0])
                input_spec_path = os.path.join(csv_dir, row[1])
                timeout = row[2]

                zono_net, input_zono, output_spec = load_net_and_input_zonotope(model_path, input_spec_path, device)
                instances.append((model_path, input_spec_path, zono_net, input_zono, output_spec))
    else:
        raise Exception("Please provide either spec or center, epsilon and true label")
    
    verified_instances = 0

    for model_path, input_spec_path, zono_net, x, output_spec in instances:
        print("Verifying network: {} with input spec: {}".format(model_path, input_spec_path))
        start_time = time.perf_counter()
        x.to(device)
        zono_net.to(device)
        if train_network(zono_net, x, output_spec):
            verified_instances += 1
        end_time = time.perf_counter()
        print("Time: {}".format(end_time - start_time))
    print("Verified instances: {}".format(verified_instances))
    print("Total instances: {}".format(len(instances)))
    print("Verified ratio: {}".format(verified_instances / len(instances)))

def train_network(net, x, output_spec):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    for i in range(10000):
        optimizer.zero_grad()
        y = net(x)
        loss = y.vnnlib_loss(output_spec)
        loss.backward()
        optimizer.step()
        if loss.item() < 0.0001:
            print("Verified!")
            print("Final loss: {}".format(loss.item()))
            print("Iterations: {}".format(i))
            return True
    print("Could not verify. Final loss: {}".format(loss.item()))
    return False


def load_net_and_input_zonotope(net_path, spec_path, device):
    num_inputs, inp_shape, num_outputs, out_shape, inp_dtype = get_num_inputs_outputs(net_path)
    torch_dtype = numpy_dtype_to_pytorch_dtype(inp_dtype)
    torch_net = convert(net_path)
    input_tensor = torch.randn(inp_shape, dtype=torch_dtype)
    zono_net = transform_network_fx(torch_net, input_tensor, optimize_alpha=True)

    spec = read_vnnlib_simple(spec_path, num_inputs, num_outputs)
    input_zono = Zonotope.from_vnnlib(spec[0][0], inp_shape, torch_dtype)

    output_specs = spec[0][1]
    output_tensors = [(torch.from_numpy(mat).to(device=device, dtype=torch_dtype),
                       torch.full(out_shape, rhs[0], dtype=torch_dtype, device=device))
                      for mat, rhs in output_specs]
    
    factors, rhs_values = zip(*output_tensors)
    factors = torch.stack(factors, dim=0)
    rhs_values = torch.stack(rhs_values, dim=0)
    output_tensors = factors, rhs_values
    return zono_net, input_zono, output_tensors


if __name__ == "__main__":
    main()
