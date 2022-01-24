# pytorch_drl
Implementation of Deep Reinforcement Learning algorithms in PyTorch, with support for distributed data collection and data-parallel training.

(In progress...)

This repo contains flexible implementations of several deep reinforcement learning algorithms.
It supports the algorithms, architectures, and rewards from several papers, including:
- [Mnih et al., 2015](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) - 'Human Level Control through Deep Reinforcement Learning'
- [van Hasselt et al., 2015](https://arxiv.org/pdf/1509.06461.pdf) - 'Deep Reinforcement Learning with Double Q-learning'
- [Wang et al., 2015](https://arxiv.org/pdf/1511.06581.pdf) - 'Dueling Network Architectures for Deep Reinforcement Learning'
- [Schaul et al., 2015](https://arxiv.org/pdf/1511.05952.pdf) - 'Prioritized Experience Replay'
- [Schulman et al., 2017](https://arxiv.org/pdf/1707.06347.pdf) - 'Proximal Policy Optimization Algorithms'
- [Burda et al., 2018](https://arxiv.org/pdf/1810.12894.pdf) - 'Exploration by Random Network Distillation'

Additionally, support is planned for:
- [Munos et al., 2016](https://arxiv.org/pdf/1606.02647.pdf) - 'Safe and Efficient Off-Policy Reinforcement Learning'
- [Pathak et al., 2017](https://arxiv.org/pdf/1705.05363.pdf) - 'Curiosity-driven Exploration by Self-supervised Prediction'
- [Haarnoja et al., 2018](https://arxiv.org/pdf/1801.01290.pdf) - 'Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor'
- [Horgan et al., 2018](https://arxiv.org/pdf/1803.00933.pdf) - 'Distributed Prioritized Experience Replay'
- [Kapturowski et al., 2018](https://openreview.net/pdf?id=r1lyTjAqYX) - 'Recurrent Experience Replay in Distributed Reinforcement Learning'
- [Badia et al., 2020](https://arxiv.org/pdf/2002.06038.pdf) - 'Never Give Up: Learning Directed Exploration Strategies'

## Getting Started

Install the following system dependencies:
#### Ubuntu
```bash
sudo apt-get update
sudo apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake libjpeg-dev zlib1g zlib1g-dev swig python3-dev
```

#### Mac OS X
Installation of the system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew update
brew install cmake
```

#### Everyone
We recommend creating a conda environment for this project. You can download the miniconda package manager from https://docs.conda.io/en/latest/miniconda.html.
Then you can set up the new conda environment as follows:

```bash
conda create --name pytorch_drl python=3.9.2
conda activate pytorch_drl
git clone https://github.com/lucaslingle/pytorch_drl
cd pytorch_drl
pip install -e .
```

## Overview

### Structure

This repo comes in two parts: a python package and a script. 

### Script

To use the script correctly, you can refer to the script usage [docs](script_usage.md). 

### Config

A formal description of how the configuration files are structured can be found in the config usage [docs](config_usage.md).  
Some example config files for different algorithms are also provided in the subdirectories of ```models_dir```.  

## Reproducing Papers

In this section, we describe the algorithms whose published results we've replicated using our codebase.

### Proximal Policy Optimization Algorithms

| Game           | OpenAI Baselines | Schulman et al., 2017 | Ours         |
| :------------- | ---------------: | --------------------: | -----------: |
| Beamrider      |          1299.3  |                1590.0 |       3406.5 |
| Breakout       |           114.3  |                 274.8 |        424.4 |
| Enduro         |           350.2  |                 758.3 |        749.4 |
| Pong           |            13.7  |                  20.7 |         19.8 |
| Qbert          |          7012.1  |               14293.3 |      16600.8 |
| Seaquest       |          1218.9  |                1204.5 |        938.00 seed 0 948.20 seed 1 |
| Space Invaders |           557.3  |                 942.5 |       1151.9 |

- For computational efficiency, we tested only the seven Atari games first examined by [Mnih et al., 2013](https://arxiv.org/pdf/1312.5602.pdf).
- For consistency with Schulman et al., 2017, each of our results above is the mean performance over the last 100 real episodes of training, averaged over three random seeds.
- The OpenAI baselines results were obtained [here](https://htmlpreview.github.io/?https://github.com/openai/baselines/blob/master/benchmarks_atari10M.htm).
- As can be seen above, our implementation closely reproduces the results of the paper. 
