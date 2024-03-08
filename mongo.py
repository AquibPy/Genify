from pymongo import MongoClient,errors
from bson import ObjectId
import os

class MongoDB():
    def __init__(self):
        self.username = os.getenv("MONGO_USERNAME")
        self.password = os.getenv("MONGO_PASSWORD")
        self.dbname = os.getenv("MONGO_DBNAME")
        self.collection_name = os.getenv("MONGO_COLLECTION")
        try:
            self.client = MongoClient(f"mongodb+srv://{self.username}:{self.password}@cluster0.sdx7i86.mongodb.net/{self.dbname}") ## i am using mongodb atlas. Use can use your local mongodb
            database = self.dbname
            collection = self.collection_name
            cursor = self.client[database]
            self.collection = cursor[collection]
        except errors.ConnectionFailure as e:
            print(f"Error: {e}")
    
    def insert_data(self,data):
        try:
            new_data =  data['Document']
            response = self.collection.insert_one(new_data)
            output = {
                "Status": "Successfully Inserted!!!",
                "Document_id": str(response.inserted_id)
            }
            return output
        except Exception as e:
            print(f"Error: {e}")
            return {'Status': 'Insertion failed.'}
    
    def read_by_id(self, id):
        try:
            document = self.collection.find_one({'_id': ObjectId(id)})
            if document:
                output = {item: document[item] for item in document if item != '_id'}
            else:
                output = {'Status': 'Document not found.'}
            return output
    
        except Exception as e:
            print(f"Error: {e}")
            return {'Status': 'Reading failed.'} 
    
    def update_data(self, id,key, value):
        try:
            filter = {"_id": ObjectId(id)}
            updated_data = {"$set": {key: value}}
            response = self.collection.update_one(filter, updated_data)
            output = {'Status': "Successfully Updated!!!" if response.modified_count > 0 else "Nothing was Updated."}
            return output
        except Exception as e:
            print(f"Error: {e}")
            return {'Status': 'Updating failed.'} 
    
    def delete_data(self,data):
        try:
            filter = data['Filter']
            response = self.collection.delete_one(filter)
            output = {'Status': 'Successfully Deleted!!!' if response.deleted_count > 0 else "Document not found."}
            return output
        except Exception as e:
            print(f"Error: {e}")
            return {'Status': 'Deleting failed.'}