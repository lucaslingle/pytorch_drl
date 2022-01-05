# pytorch_drl
Implementation of Deep Reinforcement Learning algorithms in PyTorch, with support for distributed data collection and data-parallel training.

(In progress...)

This repo contains flexible implementations of several deep reinforcement learning algorithms.
It supports the algorithms, architectures, and rewards from several papers, including:
- [Mnih et al., 2015](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) - 'Human Level Control through Deep Reinforcement Learning'
- [van Hasselt et al., 2016](https://arxiv.org/pdf/1509.06461.pdf) - 'Deep Reinforcement Learning with Double Q-learning'
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
sudo apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig python3-dev
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
- create a ```models_dir``` directory if it does not exist;
- create a subdirectory of the ```models_dir``` directory, with a descriptive name for the training run;
- copy over a config file and edit the parameters appropriately;
- pass the details to the script and run it: 
  ```
  python -m script --models_dir=YOUR_MODELS_DIR --run_name=YOUR_RUN_NAME
  ```
To evaluate a trained model, append ```--mode=eval``` to the command above.


### Config usage
The config file has several components. We detail their usage below. 
- Distributed communication:
  - The script currently only supports single-machine training with zero or more GPUs. 
  - If using GPUs, be sure to ```backend``` parameter to ```nccl```, and set the ```world_size``` to the number of GPUs available.
- Algorithm:
  - The class names of available algorithm can be found in the submodules of ```drl/algos/```.
  - Currently supported class names are: ```DQN, PPO```.
  - Depending on the algorithm, different parameter names and their values must be supplied. To see example config files for each implemented algorithm, refer to the ```models_dir``` subdirectories in this repo.
  - If any intrinsic rewards are used (discussed later), we require a field ```reward_weights``` to be included. Its value should be a dictionary of positive floats, keyed by reward name.
- Environment:
  - We allow training on any OpenAI gym environment.
  - The environment name should be specified by setting the ```id``` parameter.
- Wrappers:
  - We allow a flexible pipeline of wrappers to be applied, similar to the OpenAI gym and baselines libraries. 
  - The class names for these wrappers serve as keys in a dictionary, and their values are class-specific arguments.
  - The class names of available wrappers can be found in the submodules of ```drl/envs/wrappers/```.
- Intrinsic rewards:
  - Intrinsic rewards can be included by specifying wrappers in ```drl/envs/wrappers/stateful/intrinsic/```. 
  - The name of each reward stream is determined by the property ```reward_name``` in the wrapper it came from. 
  - Multiple intrinsic rewards can easily be combined by listing multiple wrappers.
  - In our algorithms, we follow the practice of Burda et al., 2018, Badia et al., 2020 and perform credit assignment and value estimation for each reward stream separately.
- Networks:
  - We require algorithm-specific strings corresponding to network names to be listed out as second-level under the ```networks``` heading.
  - For example, PPO potentially uses two networks: a ```policy_net``` and a ```value_net```. To avoid ambiguity arising from misspecified config files, our PPO implementation requires that if the network architecture is to be shared, the parameter under ```use_shared_architecture``` should be set to ```True```. However, both networks must be listed. We follow a similar pattern for all algorithms: all network names expected by the algorithm must be listed.
  - Under each network name should be either ```use_shared_architecture``` parameter (set to True), or else five different arguments.
- Preprocessing:
  - The first of the five arguments under each network is ```preprocessing```.
  - The class names for these preprocessing operations serve as keys in a dictionary, and their values are class-specific arguments.
  - Currently, you should only have to use ```ToChannelMajor``` (for environments with visual observations) or {}, the dictionary with no elements.
  - In general, preprocessing class names can be found in submodules of ```drl/agents/preprocessing```.  
- Architecture:
  - The second of the five arguments under each network is ```architecture```.
  - Under this heading, two values must be specified: cls_name and cls_args.
  - The class names can be found in the submodules of ```drl/agents/architectures```.
- Predictors:
  - The third of the five arguments under each network is ```predictors```.
  - We require algorithm-specific predictor names. The names required are a subset of ```{'policy', 'value_extrinsic', 'action_value_extrinsic'}```.
  - If intrinsic rewards are used, we additionally require value or action-value heads for each reward stream.
  - The names of predictor must be ```value_{reward_name}``` or ```action_value_{reward_name}```. 
- Optimizer: 
  - The fourth of the five arguments under each network is ```optimizer```.
  - Under this heading, two values must be specified: cls_name and cls_args.
  - The class names can be any optimizer in ```torch.optim```.
  - The cls_args parameter should be specified as a dictionary, whose values depend on the optimizer used.
- Scheduler:
  - The fifth of the five arguments under each network is ```scheduler```.
  - Under this heading, two values must be specified: cls_name and cls_args.
  - The class names can be any scheduler in ```torch.optim.lr_schedulers```, except ```ReduceLROnPlateau``` which requires access to a stream of loss values and is not currently supported by our algorithms. 
  - The cls_args parameter should be specified as a dictionary, whose values depend on the scheduler used.
  - The cls_name for the scheduler can also be ```None```.
    
We reiterate that some example ```config.yaml``` files are provided to help you get started.  
