"""Script."""

import argparse
import importlib
import os

import torch as tc

from drl.utils.configuration import ConfigParser


def create_argparser():
    parser = argparse.ArgumentParser(
        description="A Pytorch implementation of Deep Residual Networks, " +
                    "using Torch Distributed Data Parallel.")

    parser.add_argument("--mode", choices=['train', 'eval'], default='train')
    parser.add_argument("--models_dir", type=str, default='models_dir')
    parser.add_argument("--run_name", type=str, default='good_hparams')
    return parser


def get_config(args):
    base_path = os.path.join(args.models_dir, args.run_name)
    config_path = os.path.join(base_path, 'config.yaml')
    checkpoint_dir = os.path.join(base_path, 'checkpoints')
    log_dir = os.path.join(base_path, 'tensorboard_logs')

    config = ConfigParser(
        defaults={
            'checkpoint_dir': checkpoint_dir,
            'log_dir': log_dir
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
    if rank == 0:
        print(f"Test metrics: {metrics}")
    cleanup()


if __name__ == '__main__':
    args = create_argparser().parse_args()
    config = get_config(args)
    tc.multiprocessing.spawn(
        train if args.mode == 'train' else evaluate,
        args=(config,),
        nprocs=config['distributed']['world_size'],
        join=True)
