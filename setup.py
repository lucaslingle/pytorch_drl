from setuptools import setup


setup(
    name="pytorch_drl",
    py_modules=["drl"],
    version="0.0.2",
    description="Pytorch library implementing Deep RL algorithms.",
    author="Lucas D. Lingle",
    install_requires=[
        'torch==1.10.1',
        'torchvision==0.11.2',
        'tensorboard==2.7.0',
        'pyyaml==6.0',
        'atari-py==0.2.6',
        'gym==0.18.0',
        'moviepy==1.0.3',
        'opencv-python==4.5.2.52',
    ]
)