# Text example

This example implements training of a discrete flow matching model on text data. This repository provides the necessary tools and scripts to train and evaluate these models.

**Note:** this example was tested only using PyTorch 2.5 and on a single node of H100 (8 gpus). With this setup, we achieved approximately 380k training steps in 24 hours.

## Installation

To get started with this project, follow these steps to set up your environment:

```bash
conda env create -f environment.yml
conda activate discrete_flow_matching
```

## Usage

To train a discrete flow matching model on fine-web-edu, run:

```bash
CACHE_DIR=...

python run_train.py data.cache_dir=${CACHE_DIR}
```

To use `slurm`, modify the `slurm` config according to the cluster you are working on, and run:
```bash
CACHE_DIR=...
HYDRA_RUN_DIR=...

python run_train.py data.cache_dir=${CACHE_DIR} hydra_dir=${HYDRA_RUN_DIR} -m &
```

## Results

We trained models with linear scheduler (`PolynomialConvexScheduler(n=1.0)`) for one million steps on FineWeb-EDU.

```bash
PYTHONPATH="." python scripts/run_eval.py --work_dir "/path/to/exp/folder" --ngpus 8 --eval_elbo --eval_perplexity
```

<table>
    <thead>
        <tr>
            <th>Scheduler</th>
            <th>Source distribution</th>
            <th>Loss</th>
            <th>Generative perplexity</th>
            <th>ELBO</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>Linear</td>
            <td rowspan=2>Mask</td>
            <td>Cross-entropy</td>
            <td><center>128.9</center></td>
            <td><center>53.2</center></td>
        </tr>
        <tr>
            <td>Generalized KL</td>
            <td><center>132.2</center></td>
            <td><center>47.9</center></td>
        </tr>
        <tr>
            <td rowspan=2>Uniform</td>
            <td>Cross-entropy</td>
            <td><center>90.9</center></td>
            <td><center>71.7</center></td>
        </tr>
        <tr>
            <td>Generalized KL</td>
            <td><center>82.1</center></td>
            <td><center>71.3</center></td>
        </tr>
    </tbody>
</table>

## Folder structure

```bash
.
├── configs        # Train configs
│   └── ...
├── data           # Data loading and preprocessing
│   └── ...
├── logic          # Logic components, such as flow related classes
│   └── ...
├── model          # Transformer implementation
│   └── ...
├── scripts        # Evaluation script
│   └── ...
├── utils          # Utility functions
│    └── ...
├── README.md
├── environment.yml
├── train.py
└── run_train.py   # Run training script
```

## Implemented methods

This repository implements the following papers:
- [Discrete Flow Matching](https://arxiv.org/abs/2407.15595)
- [Flow Matching with General Discrete Paths: A Kinetic-Optimal Perspective](https://arxiv.org/abs/2412.03487)
- [Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design](https://arxiv.org/abs/2402.04997)
- [Simplified and Generalized Masked Diffusion for Discrete Data](https://arxiv.org/abs/2406.04329)


## Acknowledgements

This example partially use code from:
- [Flash attention](https://github.com/Dao-AILab/flash-attention)
- [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)
- [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://github.com/openai/glide-text2im/)
- [TorchData](https://github.com/pytorch/data/tree/main)

## License

The majority of the code in this example is licensed under CC-BY-NC, however portions of the project are available under separate license terms: 
- flash attention and TorchData are under BSD 3 license.
- Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution and GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models are under MIT license.