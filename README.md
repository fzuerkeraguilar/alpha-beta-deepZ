# abZono

abZono is a Python library for verfying properties of a neural network using the zonotope abstract domain.

## Table of Contents

- [abZono](#abzono)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)

## Installation

Clone the repository with:

```bash
git clone --recurse-submodules
```

Install the required packages with:

```bash
pip install -r requirements.txt
```

Unzip the wanted network and vnnlib files in the `test/vnnlib_202{2,3}` folder.
For example:

```bash
gunzip test/vnncomp2022_benchmarks/benchmarks/mnist_fc/oxxn/mnist-net_256x2.onnx.gz
gunzip test/vnncomp2022_benchmarks/benchmarks/mnist_fc/vnnlbi/prop_0_0.03.vnnlib.gz
```

## Usage

```bash
python -m abZono --net test/vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx --spec test/vnncomp2022_benchmarks/benchmarks/mnist_fc/vnnlib/prop_0_0.03.vnnlib
```