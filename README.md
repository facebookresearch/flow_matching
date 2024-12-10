<div align="center">

# Flow Matching

[![CI](https://github.com/facebookresearch/flow_matching/actions/workflows/ci.yaml/badge.svg)](https://github.com/facebookresearch/flow_matching/actions/workflows/ci.yaml)
[![Coverage](https://github.com/facebookresearch/flow_matching/raw/refs/heads/gh-pages/coverage/coverage-badge.svg)](https://stunning-potato-4k4z71e.pages.github.io/coverage/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

</div>

`flow_matching` is a PyTorch library for Flow Matching algorithms, featuring continuous and discrete implementations. It includes examples for both text and image modalities. This repository is part of [Flow Matching Guide and Codebase](https://arxiv.org/abs/2412.06264).


![](./assets/teaser.png)

## Installation

This repository requires Python 3.9 and Pytorch 2.1 or greater. To install the latest version run:
```
pip install flow_matching
```

## Repository structure

The core and example folders are structured in the following way:
```bash
.
├── flow_matching                  # Core library
│   ├── loss                       # Loss functions
│   │   └── ...
│   ├── path                       # Path and schedulers
│   │   ├── ...
│   │   └── scheduler              # Schedulers and transformations
│   │       └── ...
│   ├── solver                     # Solvers for continuous and discrete flows
│   │   └── ...
│   └── utils
│       └── ...
└── examples                       # Synthetic, image, and text examples
    ├── ...
    ├── image
    │       └── ...
    └── text 
            └── ...
```

## Development

To create a conda environment with all required dependencies, run:
```
conda env create -f environment.yml
conda activate flow_matching
```

Install pre-commit hook. This will ensure that all linting is done on each commit
```
pre-commit install
```

Install the `flow_matching` package in an editable mode:
```
pip install -e .
```

## FAQ

#### I want to train a Flow Matching model, where can I find the training code?

We provide [training examples](examples). Under this folder, you can find synthetic data for [continuous](examples/2d_flow_matching.ipynb), [discrete](examples/2d_discrete_flow_matching.ipynb), and [Riemannian](examples/2d_riemannian_flow_matching_flat_torus.ipynb) Flow Matching. We also provide full training [examples](examples/image) (continuous and discrete) on CIFAR10 and face-blurred ImageNet, and a scalable discrete Flow Matching example for [text modeling](examples/text).

#### Do you release pre-trained models?

In this version, we don't release pre-trained models. All models under [examples](examples) can be trained from scratch by a single running command. 

#### How to contribute to this codebase?
Please follow the [contribution guide](CONTRIBUTING.md).

## License

The code in this repository is CC BY-NC licensed. See the [LICENSE](LICENSE) for details.

## Citation

If you found this repository useful, please cite the following.

```
@misc{lipman2024flowmatchingguidecode,
      title={Flow Matching Guide and Code}, 
      author={Yaron Lipman and Marton Havasi and Peter Holderrieth and Neta Shaul and Matt Le and Brian Karrer and Ricky T. Q. Chen and David Lopez-Paz and Heli Ben-Hamu and Itai Gat},
      year={2024},
      eprint={2412.06264},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.06264}, 
}
```
