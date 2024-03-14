import argparse
import logging
from .example_vnnlib import get_num_inputs_outputs, read_vnnlib_simple
import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
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
parser.add_argument('--zono-spec', type=str, metavar='N',
                    help='Path to zono file')
parser.add_argument('--epsilon', type=float, default=0.1, help="epsilon")
parser.add_argument('--dataset', type=str, help="Dataset to use")
parser.add_argument('--subset', type=int, help="Subset of dataset to use")
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

    if args.dataset:
        instances = load_net_and_dataset(args.net, args.dataset, args.subset ,args.epsilon, device)
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
                instances.append((zono_net, input_zono, output_spec))
    else:
        raise Exception("Please provide either csv file or spec file.")

    verified_instances = 0
    sat = []
    unsat = []
    total_start_time = time.perf_counter()
    for i, (zono_net, x, output_spec) in enumerate(instances):
        start_time = time.perf_counter()
        x.to(device)
        zono_net.to(device)
        if vnnlib_train_network(zono_net, x, output_spec):
            sat += [i]
            verified_instances += 1
        else:
            unsat += [i]
        end_time = time.perf_counter()
        print("Time: {}".format(end_time - start_time))
    total_end_time = time.perf_counter()
    print("Total time: {}".format(total_end_time - total_start_time))
    print("Verified instances: {}".format(verified_instances))
    print("Total instances: {}".format(len(instances)))
    print("Verified ratio: {}".format(verified_instances / len(instances)))
    print("Sat: {}".format(sat))
    print("Unsat: {}".format(unsat))


def vnnlib_train_network(net, x, output_spec):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    for i in range(1000):
        optimizer.zero_grad()
        y = net(x)
        loss = y.vnnlib_loss(output_spec)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Loss: {}".format(loss))
        if loss.item() < 0.0001:
            print("sat")
            print("Final loss: {}".format(loss.item()))
            print("Iterations: {}".format(i))
            print("Upper bound: {}".format(y.upper_bound.tolist()))
            print("Lower bound: {}".format(y.lower_bound.tolist()))
            return True
    print("unsat")
    print("Final loss: {}".format(loss.item()))
    print("Upper bound: {}".format(y.upper_bound))
    print("Lower bound: {}".format(y.lower_bound))
    return False


def label_train_network(net, x, true_label):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    for i in range(1000):
        optimizer.zero_grad()
        y = net(x)
        loss = y.label_loss(true_label)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Loss: {}".format(loss))
        if loss.item() < 0.0001:
            print("Verified!")
            print("Final loss: {}".format(loss.item()))
            print("Iterations: {}".format(i))
            print("Upper bound: {}".format(y.upper_bound))
            print("Lower bound: {}".format(y.lower_bound))
            return True
    print("Could not verify. Final loss: {}".format(loss.item()))
    print("Upper bound: {}".format(y.upper_bound))
    print("Lower bound: {}".format(y.lower_bound))
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
                       torch.from_numpy(rhs).to(device=device, dtype=torch_dtype))
                      for mat, rhs in output_specs]

    factors, rhs_values = zip(*output_tensors)
    factors = torch.stack(factors, dim=0)
    rhs_values = torch.stack(rhs_values, dim=0)
    output_tensors = factors, rhs_values, True
    return zono_net, input_zono, output_tensors


def load_net_and_dataset(net_path, dataset, subset, epsilon, device):
    num_inputs, inp_shape, num_outputs, out_shape, inp_dtype = get_num_inputs_outputs(net_path)
    torch_dtype = numpy_dtype_to_pytorch_dtype(inp_dtype)
    torch_net = convert(net_path)
    input_tensor = torch.randn(inp_shape, dtype=torch_dtype)
    zono_net = transform_network_fx(torch_net, input_tensor, optimize_alpha=True)
    zono_net.to(device)
    instances = []

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.0,), (1,))
    ])

    if dataset == "MNIST" or dataset == "mnist":
        dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset == "CIFAR10" or dataset == "cifar10":
        dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise Exception("Dataset not supported")
    
    seleted_dataset = []

    if subset is not None:
        for i in range(1000):
            seleted_dataset.append(dataset[subset*1000 + i])
    else:
        seleted_dataset = dataset
    for images, label in seleted_dataset:
        images = images.to(device)
        output_matrix = []
        for i in range(num_outputs):
            zeros = torch.zeros(out_shape, device=device, dtype=torch_dtype)
            zeros[:, label] = -1
            zeros[:, i] = 1
            output_matrix.append(zeros)
        output_matrix = torch.stack(output_matrix, dim=0)
        output_spec = (output_matrix, torch.zeros(num_outputs, device=device, dtype=torch_dtype), False)
        zonotope = Zonotope.from_l_inf(images, epsilon, shape=torch.Size(inp_shape), l=0, u=1)
        instances.append((zono_net, zonotope, output_spec))

    return instances


if __name__ == "__main__":
    main()
