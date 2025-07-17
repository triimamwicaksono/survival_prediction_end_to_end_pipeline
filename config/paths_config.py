import os


RAW_DIR = "artifacts/raw"
PROCESSED_DIR = "artifacts/processed"
TRAIN_PATH = os.path.join(RAW_DIR,'titanic_train.csv')
TEST_PATH = os.path.join(RAW_DIR,'titanic_test.csv')
MODEL_OUTPUT_PATH = "artifacts/models/random_forest_model.pkl"