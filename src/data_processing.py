from src.feature_store import RedisFeatureStore
import os
from src.logger import get_logger
from src.custom_exception import CustomException
import pandas as pd
from config.paths_config import TRAIN_PATH, TEST_PATH
from imblearn.over_sampling import SMOTE


logger = get_logger(__name__)

class DataProcessing:
    
    def __init__(self, feature_store: RedisFeatureStore, train_path, test_path):
        self.feature_store = feature_store
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_resampled = None
        self.y_resampled = None

    def load_data(self):
        try:
            self.train_data = pd.read_csv(self.train_path)
            self.test_data = pd.read_csv(self.test_path)
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException(f"Data loading failed: {e}")
    
    def preprocess_data(self):
        try:
            self.train_data['Age'] = self.train_data['Age'].fillna(self.train_data['Age'].median())
            self.train_data['Embarked'] = self.train_data['Embarked'].fillna(self.train_data['Embarked'].mode()[0])
            self.train_data['Fare'] = self.train_data['Fare'].fillna(self.train_data['Fare'].median())
            self.train_data['Sex'] = self.train_data['Sex'].map({'male': 0, 'female': 1})
            self.train_data['Embarked'] = self.train_data['Embarked'].astype('category').cat.codes
            self.train_data['Familysize'] = self.train_data['SibSp'] + self.train_data['Parch'] + 1
            self.train_data['Isalone'] = (self.train_data['Familysize'] == 1).astype(int)
            self.train_data['HasCabin'] = self.train_data['Cabin'].notnull().astype(int)
            self.train_data['Title'] = self.train_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False).map(
                {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
            ).fillna(4)
            self.train_data['Pclass_Fare'] = self.train_data['Pclass'] * self.train_data['Fare']
            self.train_data['Age_Fare'] = self.train_data['Age'] * self.train_data['Fare']

            logger.info("Data preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise CustomException(f"Data preprocessing failed: {e}")
    
    def handle_imbalanced_data(self):
        try:
            X = self.train_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Familysize', 'Isalone', 'HasCabin', 'Title', 'Pclass_Fare', 'Age_Fare']]
            y = self.train_data['Survived']
            smote = SMOTE(random_state=42)
            self.X_resampled, self.y_resampled = smote.fit_resample(X, y)
            logger.info("Imbalanced data handled successfully")
        except Exception as e:
            logger.error(f"Error handling imbalanced data: {e}")
            raise CustomException(f"Imbalanced data handling failed: {e}")
        
    def store_features(self):
        try:
            batch_data = {}
            for index, row in self.train_data.iterrows():
                entity_id = row["PassengerId"]
                features = {
                    "Age" : row['Age'],
                    "Fare" : row["Fare"],
                    "Pclass" : row["Pclass"],
                    "Sex" : row["Sex"],
                    "Embarked" : row["Embarked"],
                    "Familysize": row["Familysize"],
                    "Isalone" : row["Isalone"],
                    "HasCabin" : row["HasCabin"],
                    "Title" : row["Title"],
                    "Pclass_Fare" : row["Pclass_Fare"],
                    "Age_Fare" : row["Age_Fare"],
                    "Survived" : row["Survived"]
                }
                batch_data[entity_id] = features
            self.feature_store.store_batch_features(batch_data)
            logger.info("Data has been feeded into Feature Store..")
        except Exception as e:
            logger.error(f"Error while feature storing data {e}")
            raise CustomException(str(e))
        
    def retrieve_features(self,entity_id):
        try:
            features = self.feature_store.get_features(entity_id)
            if features:
                logger.info("Features retrieved successfully")
                return features
            else:
                logger.warning("No features found for the given entity IDs")
                return None
        except Exception as e:
            logger.error(f"Error retrieving features: {e}")
            raise CustomException(f"Feature retrieval failed: {e}") 
    
    def run(self):
        try:
            logger.info("Starting data processing")
            self.load_data()
            self.preprocess_data()
            self.handle_imbalanced_data()
            self.store_features()
            logger.info("Data processing completed successfully")
        except CustomException as e:
            logger.error(f"Data processing failed: {e}")
            raise e
if __name__ == "__main__":
    feature_store = RedisFeatureStore()
    data_processor = DataProcessing(feature_store, TRAIN_PATH, TEST_PATH)
    data_processor.run()

    print(data_processor.retrieve_features(entity_id=332))
