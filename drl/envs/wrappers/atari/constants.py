ACTION_RESET_SEQUENCE = [1, 2]
REWARD_CLIP_LOW = -1.0
REWARD_CLIP_HIGH = 1.0
LIVES_FN = lambda env: env.unwrapped.ale.lives()
NOOP_ACTION = 0
NUM_STACK = 4
NUM_SKIP = 4
APPLY_MAX = True
MIN_RESET_NOOPS = 30
MAX_RESET_NOOPS = 30
TARGET_HEIGHT = 84
TARGET_WIDTH = 84
USE_GRAYSCALE = True
SCALE_FACTOR = 1 / 255
