from setuptools import setup

setup(
    name="pytorch_drl",
    py_modules=["drl"],
    version="0.0.4",
    description="Pytorch library implementing Deep RL algorithms.",
    author="Lucas D. Lingle",
    install_requires=[
        'atari-py==0.2.6',
        'coverage==6.3.1',
        'gym==0.18.0',
        'moviepy==1.0.3',
        'opencv-python==4.5.2.52',
        'tensorboard==2.7.0',
        'torch==1.10.1',
        'pytest==6.2.5',
        'pyyaml==6.0',
    ])
