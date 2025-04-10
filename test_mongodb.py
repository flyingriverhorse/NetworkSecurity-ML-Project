from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv
# Replace the placeholder with your actual MongoDB connection string
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
# Load environment variables from .env file
uri = MONGO_DB_URL

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)