"""Script."""

import argparse
import importlib
import os

import torch as tc

from drl.utils.configuration import ConfigParser


def create_argparser():
    parser = argparse.ArgumentParser(
        description='Deep RL algorithms, using Torch DDP.')
    parser.add_argument(
        '--mode', choices=['train', 'evaluate', 'video'], default='train')
    parser.add_argument(
        '--models_dir', type=str, default='models_dir')
    parser.add_argument(
        '--experiment_group', type=str, default='my_old_repo_ppo_hparams')
    parser.add_argument(
        '--env_name', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument(
        '--seed', type=int, default=0)
    return parser


def get_config(args):
    experiment_group_path = os.path.join(args.models_dir, args.experiment_group)
    config_path = os.path.join(experiment_group_path, 'config.yaml')

    experiment_path = os.path.join(
        experiment_group_path, args.env_name, str(args.seed))
    checkpoint_dir = os.path.join(experiment_path, 'checkpoints')
    log_dir = os.path.join(experiment_path, 'tensorboard_logs')

    config = ConfigParser(
        defaults={
            'checkpoint_dir': checkpoint_dir,
            'log_dir': log_dir,
            'seed': args.seed
        }
    )
    config.read(config_path, verbose=True)
    return config


def get_algo(rank, config):
    module = importlib.import_module('drl.algos')
    algo = getattr(module, config.get('algo').get('cls_name'))
    return algo(rank, config)


def setup(rank, config):
    distributed_config = config.get('distributed')
    os.environ['MASTER_ADDR'] = distributed_config.get('master_addr')
    os.environ['MASTER_PORT'] = distributed_config.get('master_port')
    tc.distributed.init_process_group(
        backend=distributed_config.get('backend'),
        world_size=distributed_config.get('world_size'),
        rank=rank)
    algo = get_algo(rank, config)
    return algo


def cleanup():
    tc.distributed.destroy_process_group()


def train(rank, config):
    algo = setup(rank, config)
    algo.training_loop()
    cleanup()


def evaluate(rank, config):
    algo = setup(rank, config)
    metrics = algo.evaluation_loop()
    if algo.rank == 0:
        print(f"Test metrics: {metrics}")
    cleanup()


def video(rank, config):
    algo = setup(rank, config)
    _ = algo.render_loop()
    cleanup()


if __name__ == '__main__':
    args = create_argparser().parse_args()
    config = get_config(args)
    ops = {
        'train': train,
        'evaluate': evaluate,
        'video': video
    }
    tc.multiprocessing.spawn(
        ops[args.mode],
        args=(config,),
        nprocs=config['distributed']['world_size'],
        join=True)
