import gym

from drl.envs.wrappers.common import (
    NoopResetWrapper,
    MaxAndSkipWrapper,
    TimeLimitWrapper,
    EpisodicLifeWrapper,
    ActionResetWrapper,
    ResizeWrapper,
    ScaleObservationsWrapper,
    ClipRewardWrapper,
    FrameStackWrapper
)
from drl.envs.atari.constants import (
    ACTION_RESET_SEQUENCE,
    REWARD_CLIP_LOW,
    REWARD_CLIP_HIGH,
    LIVES_FN,
    NOOP_ACTION,
    NUM_STACK,
    NUM_SKIP,
    APPLY_MAX,
    MIN_RESET_NOOPS,
    MAX_RESET_NOOPS,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    USE_GRAYSCALE,
    SCALE_FACTOR,
)


def make_atari(env, max_episode_steps=None):
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetWrapper(
        env=env,
        noop_action=NOOP_ACTION,
        noop_min=MIN_RESET_NOOPS,
        noop_max=MAX_RESET_NOOPS)
    env = MaxAndSkipWrapper(
        env=env,
        num_skip=NUM_SKIP,
        apply_max=APPLY_MAX)
    if max_episode_steps is not None:
        env = TimeLimitWrapper(
            env=env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind(env, episode_life=True, clip_rewards=True, scale=True, frame_stack=True):  # ppo sets frame_stack to false O_o
    """
    Configure environment for DeepMind-style Atari.

    :param env (Env): OpenAI gym environment instance.
    :param episode_life (bool): Use EpisodicLifeWrapper?
    :param clip_rewards (bool): Use ClipRewardWrapper?
    :param scale (bool): Use ScaleObservationsWrapper?
    :param frame_stack (bool): Use FrameStackWrapper?
    :return (Env): Wrapped environment.
    """
    env = ResizeWrapper(
        env=env,
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
        grayscale=USE_GRAYSCALE)

    if episode_life:
        env = EpisodicLifeWrapper(
            env=env,
            lives_fn=LIVES_FN,
            noop_action=NOOP_ACTION)

    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = ActionResetWrapper(
            env=env,
            action_sequence=ACTION_RESET_SEQUENCE)

    if clip_rewards:
        env = ClipRewardWrapper(
            env=env,
            low=REWARD_CLIP_LOW,
            high=REWARD_CLIP_HIGH)

    if scale:
        env = ScaleObservationsWrapper(
            env=env,
            scale_factor=SCALE_FACTOR)

    if frame_stack:
        env = FrameStackWrapper(
            env=env,
            num_stack=NUM_STACK)

    return env