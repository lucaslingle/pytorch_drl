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

This repo comes in two parts: a python package and a script. The script organizes all runs in a ```models_dir```, placing checkpoints and tensorboard logs in a ```run_name``` subdirectory. 

Furthermore, it expects to find a ```config.yaml``` file in the ```run_name``` directory, specifying hyperparameters and configuration details for the ```run_name``` training run. 

Using a flexible markup language like YAML allows us to specify nested configuration details, which simplifies the implementation over that of a 'flat' configuration object like an argparse namespace. 

## Usage

### Script overview
To train a new model, you should:
- create a subdirectory of the ```models_dir``` directory, with a descriptive name for the training run;
- copy over a config file and edit the parameters appropriately;
- pass the details to the script and run it: 
  ```
  python -m script --run_name=YOUR_RUN_NAME
  ```
To evaluate a trained model, append ```--mode=eval``` to the command above.

### Config overview

Some example config files for different algorithms are provided in the subdirectories of ```models_dir```.  
A formal description of how the configuration files are structured can be found in [this document](config_usage.md).  
