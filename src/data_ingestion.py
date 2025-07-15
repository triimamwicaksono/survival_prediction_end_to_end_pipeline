import os
from src.logger import get_logger
from src.custom_exception import CustomException
import pandas as pd
from config.paths_config import TRAIN_PATH, TEST_PATH
import psycopg2
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.username = os.getenv('username', 'postgres')
        self.password = os.getenv('password', 'postgres')
        self.db_name = 'postgres'
        self.host = os.getenv('host', 'localhost')
        self.port = os.getenv('port', '5432')

    def connect_to_db(self):
        try:
            connection = psycopg2.connect(dbname = self.db_name, user = self.username, password = self.password, host = self.host, port = self.port)
            self.logger.info("Database connection established successfully")
            return connection
        except Exception as e:
            self.logger.error(f"Error connecting to database: {e}")
            raise CustomException(f"Database connection failed: {e}")
        
    def fetch_data(self):
        query = "SELECT * FROM public.titanic"
        os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(TEST_PATH), exist_ok=True)
        try:
            cursor = self.connect_to_db().cursor()
            cursor.execute(query)
            data = cursor.fetchall()
            colnames = [desc[0] for desc in cursor.description]
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            self.logger.info("Data fetched and split into train and test sets")
            train_df = pd.DataFrame(train_data,columns=colnames)
            test_df = pd.DataFrame(test_data,columns=colnames)
            train_df.to_csv(TRAIN_PATH, index=False)
            test_df.to_csv(TEST_PATH, index=False)
            self.logger.info(f"Train data saved to {TRAIN_PATH} and test data saved to {TEST_PATH}")
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            raise CustomException(f"Data fetching failed: {e}")
        
    def run(self):
        try:
            self.logger.info("Statring data ingestion process")
            self.fetch_data()
            self.logger.info("Data ingestion process completed successfully")
        except CustomException as e:
            self.logger.error(f"Data ingestion failed: {e}")
            raise e
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.run()

                            
        
        
            
