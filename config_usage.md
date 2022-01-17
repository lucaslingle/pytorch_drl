# Config usage

The config file has several components. We detail their usage below. 

### Distributed communication:
  - The script currently only supports single-machine training without GPUs. (This is subject to change, however.)
  - The script currently does not support vectorized environments. (This is subject to change, however.) 
  - To use Torch DDP with more than one parallel process, you can set ```world_size``` to the number of environments you want to run.
  - If using GPUs, be sure to ```backend``` parameter to ```nccl```, and set the ```world_size``` to the number of GPUs available. 
### Algorithm:
  - The class names of available algorithms can be found in the submodules of ```drl/algos/```.
  - Currently supported class names are: ```DQN, PPO```.
  - Depending on the algorithm, different parameter names and their values must be supplied. To see example config files for each implemented algorithm, refer to the ```models_dir``` subdirectories in this repo.
  - If any intrinsic rewards are used (discussed later), we require a field ```reward_weights``` to be included. Its value should be a dictionary of positive floats, keyed by reward name.
### Environment:
  - We allow training on any OpenAI gym environment.
  - The environment name should be specified by setting the ```id``` parameter.
### Wrappers:
  - We allow a flexible pipeline of wrappers to be applied, similar to the OpenAI gym and baselines libraries. 
  - In the config file and its parse, the class names for these wrappers serve as keys in a dictionary, and their values are class-specific arguments.
  - The class names of available wrappers can be found in the submodules of ```drl/envs/wrappers/```.
  - Wrappers are applied in the order they are specified. 
### Intrinsic rewards:
  - Intrinsic rewards can be included by specifying wrappers in ```drl/envs/wrappers/stateful/intrinsic/```. 
  - The name of each reward stream is determined by the property ```reward_name``` in the wrapper it came from. 
  - Multiple intrinsic rewards can easily be combined by listing multiple wrappers.
  - In our algorithms, we follow the practice of Burda et al., 2018, Badia et al., 2020 and perform credit assignment and value estimation for each reward stream separately.
### Networks:
  - We require algorithm-specific strings corresponding to network names to be listed out as second-level under the ```networks``` heading.
  - For example, PPO potentially uses two networks: a ```policy_net``` and a ```value_net```. To avoid ambiguity arising from misspecified config files, our PPO implementation requires that if the network architecture is to be shared, the parameter ```use_shared_architecture``` under ```value_net``` should be set to ```True```. However, both network names must be listed. We follow a similar pattern for all algorithms; all network names expected by the algorithm must be listed. 
  - Under each network name should be either ```use_shared_architecture``` parameter (set to True), or else five different arguments.
### Preprocessing:
  - The first of the five arguments under each network is ```preprocessing```.
  - In the config file and its parse, the class names for these preprocessing operations serve as keys in a dictionary, and their values are class-specific arguments.
  - Currently, you should only have to use ```ToChannelMajor``` (for environments with visual observations) or {}, the dictionary with no elements.
  - In general, preprocessing class names can be found in submodules of ```drl/agents/preprocessing```.
  - Preprocessing operations are applied in the order they are specified. 
### Architecture:
  - The second of the five arguments under each network is ```architecture```.
  - Under this heading, two values must be specified: cls_name and cls_args.
  - The class names can be found in the submodules of ```drl/agents/architectures```.
### Predictors:
  - The third of the five arguments under each network is ```predictors```.
  - We require algorithm-specific predictor names. The names required are a subset of ```{'policy', 'value_extrinsic', 'action_value_extrinsic'}```.
  - Under each predictor name, two values must be specified: cls_name and cls_args.
  - In general, predictor class names can be found in submodules of ```drl/agents/heads```.
  - In general, all predictor class arguments must be supplied, with one exception: the field action_dim or num_actions can be omitted, as it will automatically be inferred per-environment. 
  - If intrinsic rewards are used, we additionally require value or action-value heads for each reward stream.
  - The names of the predictors associated with each intrinsic reward stream must be ```value_{reward_name}``` or ```action_value_{reward_name}```. 
### Optimizer: 
  - The fourth of the five arguments under each network is ```optimizer```.
  - Under this heading, two values must be specified: cls_name and cls_args.
  - The class names can be any optimizer in ```torch.optim```.
  - The cls_args parameter should be specified as a dictionary, whose values depend on the optimizer used.
### Scheduler:
  - The fifth of the five arguments under each network is ```scheduler```.
  - Under this heading, two values must be specified: cls_name and cls_args.
  - The class names can be any scheduler in ```torch.optim.lr_schedulers```, except ```ReduceLROnPlateau``` which requires access to a stream of loss values and is not currently supported by our algorithms.  
  - The cls_args parameter should be specified as a dictionary, whose values depend on the scheduler used.
  - The cls_name for the scheduler can also be ```None```.

We reiterate that some example ```config.yaml``` files are provided to help you get started.  
You can also learn more about YAML syntax [here](https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html).