from src.custom_exception import CustomException
from src.logger import get_logger
from src.feature_store import RedisFeatureStore
from config.paths_config import TRAIN_PATH, TEST_PATH, MODEL_OUTPUT_PATH
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
import pickle
import os 
import mlflow


logger = get_logger(__name__)
mlflow.set_tracking_uri("http://localhost:5050")

class ModelTraining:
    def __init__(self, feature_store: RedisFeatureStore, model_output_path, train_path, test_path):
        self.feature_store = feature_store
        self.model_output_path = model_output_path
        self.train_path = train_path
        self.test_path = test_path

        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

        logger.info("Model Training initialized...")
        
    def load_data_from_redis(self , entity_ids):
        try:
            logger.info("Extracting data from Redis")

            data = []
            for entity_id in entity_ids:
                features = self.feature_store.get_features(entity_id)
                if features:
                    data.append(features)
                else:
                    logger.warning(f"Feature not found for entity_id: {entity_id}")
            return data
        except Exception as e:
            logger.error(f"Error while loading data from Redis {e}")
            raise CustomException(str(e))
                
    def prepare_data(self):
        try:
            entity_ids = self.feature_store.get_all_entity_ids()
            train_entity_ids, test_entity_ids = train_test_split(entity_ids, test_size=0.2, random_state=42)

            train_data = self.load_data_from_redis(train_entity_ids)
            test_data = self.load_data_from_redis(test_entity_ids)

            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)

            X_train = train_df.drop(columns=['Survived'])
            y_train = train_df['Survived']
            X_test = test_df.drop(columns=['Survived'])
            y_test = test_df['Survived']

            logger.info("Data prepared successfully")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise CustomException(f"Data preparation failed: {e}")
        
    def hyperparameter_tuning(self, X_train, y_train):
        try:
            param_distributions = {
                                    'n_estimators': [100, 200, 300],
                                    'max_depth': [10, 20, 30],
                                    'min_samples_split': [2, 5],
                                    'min_samples_leaf': [1, 2]
                                }
            rf = RandomForestClassifier(random_state=42)
            random_search = RandomizedSearchCV(rf, param_distributions, n_iter=10, cv=3, scoring='accuracy', random_state=42)
            random_search.fit(X_train, y_train)
            logger.info(f"Best parameters found: {random_search.best_params_}")
            return random_search.best_estimator_
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {e}")
            raise CustomException(f"Hyperparameter tuning failed: {e}")
        
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        try:
            best_rf = self.hyperparameter_tuning(X_train, y_train)
            # self.model.fit(X_train, y_train)
            self.model = best_rf
            y_pred = best_rf.predict(X_test)
            accuracy= accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            self.save_model(best_rf)
            logger.info(f"Model trained successfully with accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1-score: {f1}")
            return {
                "accuracy":accuracy,
                "precision":precision,
                "recall":recall,
                "f1_score":f1
                 }
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise CustomException(f"Model training failed: {e}")
        
    def save_model(self, model):
        try:
            with open(self.model_output_path, 'wb') as model_file:
                pickle.dump(model, model_file)
                logger.info(f"Model saved to {self.model_output_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise CustomException(f"Model saving failed: {e}")  
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting model training process")

                # Log dataset files as artifacts if exist
                if os.path.exists(self.train_path):
                    mlflow.log_artifact(self.train_path, artifact_path="datasets")
                if os.path.exists(self.test_path):
                    mlflow.log_artifact(self.test_path, artifact_path="datasets")

                X_train, y_train, X_test, y_test = self.prepare_data()
                metrics = self.train_and_evaluate(X_train, y_train, X_test, y_test)
                

                # Log model artifact (pickle file)
                mlflow.log_artifact(self.model_output_path, artifact_path="models")

                # Log model parameters
                mlflow.log_params(self.model.get_params())
                # Log evaluation metrics
                mlflow.log_metrics(metrics)

                logger.info("Model training process completed successfully")
        except Exception as e:
            logger.error(f"Error while model training pipeline {e}")
            raise CustomException(str(e))


if __name__ == "__main__":
    feature_store = RedisFeatureStore()
    model_trainer = ModelTraining(feature_store, model_output_path=MODEL_OUTPUT_PATH, train_path=TRAIN_PATH, test_path=TEST_PATH)
    model_trainer.run()