"""Configuration for screen detector V3 trainer.

Two-stage CNN + FFT Branch architecture:
- Stage 1: natural vs screenshot
- Stage 2: screenshot vs screen_photo
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "input"
TRAINER_ROOT = PROJECT_ROOT / "trainer"
CHECKPOINT_DIR = TRAINER_ROOT / "checkpoints"
LOG_DIR = TRAINER_ROOT / "logs"

# Stage 1: natural vs screenshot
CLASS_NAMES_STAGE1 = ["natural", "screenshot"]
STAGE1_DATA_MAP = {
    "natural": ["natural_photo"],  # 递归扫描子目录
    "screenshot": ["screenshot", "hard_negative"],
}

# Stage 2: screenshot vs screen_photo
CLASS_NAMES_STAGE2 = ["screenshot", "screen_photo"]
STAGE2_DATA_MAP = {
    "screenshot": ["screenshot"],
    "screen_photo": ["screen_photo"],
}

# Model
MODEL_NAME = "efficientnet_b0"
IMAGE_SIZE = 224
INPUT_CHANNELS = 3
NUM_CLASSES = 2  # 每阶段都是二分类

# Training
BATCH_SIZE = 16
NUM_WORKERS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 30
TRAIN_VAL_SPLIT = 0.8
RANDOM_SEED = 42

# Augmentation
JPEG_QUALITY_RANGE = (50, 95)
BLUR_SIGMA_RANGE = (0.5, 2.0)
NOISE_STD_RANGE = (5, 25)
BRIGHTNESS_RANGE = (0.8, 1.2)
CONTRAST_RANGE = (0.8, 1.2)
