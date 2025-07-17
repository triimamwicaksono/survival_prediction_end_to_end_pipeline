import redis
import json


class RedisFeatureStore:
    def __init__(self, host="localhost", port=6379, db=0):
        self.client = redis.StrictRedis(host=host, port=port, db=db, decode_responses=True)
    
    def strore_features(self, entity_id, features):
        """
        Store features in Redis.
        
        :param entity_id: Unique key for the feature set.
        :param features: Dictionary of features to store.
        """
        key = f"entity:{entity_id}:features"
        self.client.set(key, json.dumps(features))

    def get_features(self, entity_id):
        """
        Retrieve features from Redis.
        
        :param entity_id": Unique key for the feature set.
        """
        key = f"entity:{entity_id}:features"
        features = self.client.get(key)
        if features:
            return json.loads(features)
        return None
    
    def store_batch_features(self, batch_data):
        """
        Store a batch of features in Redis
        
        :param batch_data: list of dictionaries, each containing an entity_id and its features.
        """
        for entity_id, features in batch_data.items():
            self.strore_features(entity_id, features)

    def get_batch_features(self, entity_ids):
        """"
        Retrieve a batch of features from Redis.
        
        :param entity_ids: List of unique keys for the feature sets.
        """
        batch_features = {}
        for entity_id in entity_ids:
            features = self.get_features(entity_id)
            if features:
                batch_features[entity_id] = features
            return batch_features
        
    def get_all_entity_ids(self):
        """
        Retrieve all entity IDs stored in Redis.
        
        :return: List of all entity IDs.
        """
        keys = self.client.keys("entity:*:features")
        return [key.split(":")[1] for key in keys]

