# Script usage

### Experiment Groups

The script structures all runs into experiment groups. 
To create a new experiment group, you should create a subdirectory of ```models_dir```, 
and create a file ```config.yaml``` in that subdirectory. 
See the [config usage](config_usage.md) document for futher details. 

### Training
To run a new experiment, simply run
  ```
  python -m script \
      --mode=train \
      --experiment_group=EXPERIMENT_GROUP_NAME \
      --env_name=ENV_NAME \
      --seed=RNG_SEED_INT
  ```

### Checkpointing
By default, the script looks for saved checkpoints, and will use them if you do not delete them.  
The checkpoints can be found in
```
models_dir/EXPERIMENT_GROUP_NAME/ENV_NAME/RNG_SEED_INT/checkpoints
```

### Tensorboard logs
To monitor tensorboard logs, you should type
```
tensorboard \
    --logdir=models_dir/EXPERIMENT_GROUP_NAME/ENV_NAME/RNG_SEED_INT/tensorboard_logs \
    --host=localhost
```
And then navigate to ```http://localhost:6006``` in your browser.

### Evaluation
Sometimes algorithm evaluation is conducted after training of the RL agent has ended.
To evaluate an agent that supports this type of evaluation, you can run
  ```
  python -m script \
      --mode=evaluate \
      --experiment_group=EXPERIMENT_GROUP_NAME \
      --env_name=ENV_NAME \
      --seed=RNG_SEED_INT
  ```

### Video

To see video of the agent interacting with the environment, you can run
  ```
  python -m script \
      --mode=video \
      --experiment_group=EXPERIMENT_GROUP_NAME \
      --env_name=ENV_NAME \
      --seed=RNG_SEED_INT
  ```
The video will be saved to 
```
models_dir/EXPERIMENT_GROUP_NAME/ENV_NAME/RNG_SEED_INT/media
```
