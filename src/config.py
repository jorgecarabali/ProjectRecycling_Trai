### src/config.py

import yaml 
from pathlib import Path

PROJECT_ROOT_PATH=Path(__file__).parent.parent.resolve()




CONFIG_PATH = PROJECT_ROOT_PATH / "config/config.yaml"

with open(CONFIG_PATH, 'r') as f:
   data=yaml.full_load(f)
          
# Paths
DATA_DIR= PROJECT_ROOT_PATH/data.get('data_dir')
TRAIN_DIR=PROJECT_ROOT_PATH/data.get('train_dir')
TEST_DIR=PROJECT_ROOT_PATH/data.get('test_dir')
MODEL_DIR=PROJECT_ROOT_PATH/data.get('model_dir')
TEST_DIR_CARDBOARD=PROJECT_ROOT_PATH/data.get('test_dir_carboard')
TEST_DIR_PLASTIC=PROJECT_ROOT_PATH/data.get('test_dir_plastic')
TEST_DIR_PAPER=PROJECT_ROOT_PATH/data.get('test_dir_paper')
TEST_DIR_EWASTE=PROJECT_ROOT_PATH/data.get('test_dir_ewaste')
MODEL_PATH=PROJECT_ROOT_PATH/data.get('model_path')

# Models and Data parameters
IMG_SIZE= data.get('img_size')
BATCH_SIZE=data.get ('batch_size')
VALIDATION_SPLIT = data.get('validation_split')
INPUT_SHAPE = data.get('input_shape')
BASE_LEARNING_RATE = data.get('base_learning_rate')
EPOCHS = data.get('epochs')

