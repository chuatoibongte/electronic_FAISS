import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from utils import build_description_concat
from dotenv import load_dotenv
import os

DIM = 768
MODEL_NAME = 'intfloat/multilingual-e5-base'

load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client['ElectronicMaster_DB']
collection = db['electronics']

model = SentenceTransformer(MODEL_NAME)

docs = list(collection.find({}))
mongo_ids = []
description_list = []

for doc in docs:
    description_concat = build_description_concat(doc)
    description_list.append("passage: " + description_concat)
    mongo_ids.append(str(doc['_id']))  # Lưu _id (string), không cần int

embeddings = model.encode(description_list, show_progress_bar=True, normalize_embeddings=True)

index = faiss.IndexFlatIP(DIM)
index.add(np.array(embeddings, dtype='float32'))
faiss.write_index(index, 'faiss_index.bin')

# Lưu mapping index <-> _id ra file
with open('faiss_id_map.txt', 'w', encoding='utf-8') as f:
    for _id in mongo_ids:
        f.write(_id + '\n')

print("✅ Đã build xong FAISS index và mapping _id (faiss_id_map.txt).")
