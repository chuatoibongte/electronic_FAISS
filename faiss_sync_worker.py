import numpy as np
import faiss
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
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

index = faiss.read_index('faiss_index.bin')
model = SentenceTransformer(MODEL_NAME)

MAPPING_FILE = 'faiss_id_map.txt'

def read_mapping():
    with open(MAPPING_FILE, encoding='utf-8') as f:
        return [line.strip() for line in f]

def write_mapping(mapping):
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        for _id in mapping:
            f.write(_id + '\n')

def sync_added(doc):
    description_concat = build_description_concat(doc)
    embedding = model.encode(['passage: ' + description_concat], normalize_embeddings=True)[0]
    index.add(np.array([embedding], dtype='float32'))
    faiss.write_index(index, 'faiss_index.bin')

    # Thêm _id vào cuối mapping
    mapping = read_mapping()
    mapping.append(str(doc['_id']))
    write_mapping(mapping)
    print(f"Thêm mới: {doc['_id']}")

def sync_deleted(oid):
    global index  # <-- Đặt ở đầu hàm
    mapping = read_mapping()
    oid_str = str(oid)
    if oid_str in mapping:
        pos = mapping.index(oid_str)
        # Xóa vector tại pos trong FAISS
        mask = np.ones(index.ntotal, dtype=bool)
        mask[pos] = False
        vectors = index.reconstruct_n(0, index.ntotal)
        new_vectors = vectors[mask]
        new_index = faiss.IndexFlatIP(DIM)
        new_index.add(np.array(new_vectors, dtype='float32'))
        faiss.write_index(new_index, 'faiss_index.bin')
        # Xóa _id khỏi mapping
        mapping.pop(pos)
        write_mapping(mapping)
        print(f"Xóa: {oid}")
        # Reload index vào worker
        index = new_index
    else:
        print(f"Không tìm thấy _id {oid} trong mapping. Không xóa.")

def sync_updated(doc):
    global index  # <-- Đặt ở đầu hàm
    mapping = read_mapping()
    oid_str = str(doc['_id'])
    if oid_str in mapping:
        # 1. Xóa cũ
        pos = mapping.index(oid_str)
        mask = np.ones(index.ntotal, dtype=bool)
        mask[pos] = False
        vectors = index.reconstruct_n(0, index.ntotal)
        new_vectors = vectors[mask]
        new_index = faiss.IndexFlatIP(DIM)
        new_index.add(np.array(new_vectors, dtype='float32'))
        mapping.pop(pos)
        # 2. Thêm lại embedding mới
        description_concat = build_description_concat(doc)
        embedding = model.encode(['passage: ' + description_concat], normalize_embeddings=True)[0]
        new_index.add(np.array([embedding], dtype='float32'))
        mapping.append(oid_str)
        write_mapping(mapping)
        faiss.write_index(new_index, 'faiss_index.bin')
        print(f"Cập nhật: {doc['_id']}")
        # Reload index vào worker
        index = new_index
    else:
        print(f"Không tìm thấy _id {doc['_id']} trong mapping để cập nhật.")

def main():
    print("⏳ Worker FAISS-MongoDB Sync đang chạy...")
    with collection.watch(full_document='updateLookup') as stream:
        for change in stream:
            if change['operationType'] == 'insert':
                doc = change['fullDocument']
                sync_added(doc)
            elif change['operationType'] in ['update', 'replace']:
                doc = change['fullDocument']
                sync_updated(doc)
            elif change['operationType'] == 'delete':
                oid = change['documentKey']['_id']
                sync_deleted(oid)

if __name__ == '__main__':
    main()
