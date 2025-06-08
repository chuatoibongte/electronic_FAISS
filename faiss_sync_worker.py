import numpy as np
import faiss
import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from transformers import AutoImageProcessor, SwinModel
from utils import build_description_concat, get_mean_image_embedding

# ========== CONFIG ==========
DIM_TEXT = 768
MODEL_NAME_TEXT = 'intfloat/multilingual-e5-base'
DIM_IMAGE = 1024  # Swin-base
MODEL_NAME_IMAGE = "microsoft/swin-base-patch4-window7-224"
MAPPING_FILE = 'faiss_id_map.txt'

# ========== LOAD ==========
load_dotenv()
MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client['ElectronicMaster_DB']
collection = db['electronics']

index_text = faiss.read_index('faiss_index.bin')
index_image = faiss.read_index('faiss_index_image.bin')
model_text = SentenceTransformer(MODEL_NAME_TEXT)
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME_IMAGE)
swin_model = SwinModel.from_pretrained(MODEL_NAME_IMAGE)

def read_mapping():
    if not os.path.exists(MAPPING_FILE):
        return []
    with open(MAPPING_FILE, encoding='utf-8') as f:
        return [line.strip() for line in f]

def write_mapping(mapping):
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        for _id in mapping:
            f.write(_id + '\n')

def try_get_image_embedding(image_urls, swin_model, image_processor, max_retry=5):
    for attempt in range(max_retry):
        mean_vec = get_mean_image_embedding(image_urls, swin_model, image_processor)
        if mean_vec is not None:
            if attempt > 0:
                print(f"✅ Đã lấy được image embedding ở lần thử thứ {attempt+1}")
            return mean_vec
        print(f"⏳ Thử lại lần {attempt+1}/{max_retry} với image...")
    print("⛔️ Không lấy được image embedding sau 5 lần thử.")
    return None

def sync_added(doc):
    # TEXT
    description_concat = build_description_concat(doc)
    embedding_text = model_text.encode(['passage: ' + description_concat], normalize_embeddings=True)[0]
    # IMAGE (thử lại tối đa 5 lần)
    image_urls = [img['url'] for img in doc.get('electronicImgs', []) if img.get('url')]
    mean_vec = try_get_image_embedding(image_urls, swin_model, image_processor, max_retry=5)
    if mean_vec is None:
        print(f"⚠️ Không thêm được IMAGE cho: {doc['_id']} (skip luôn cả text/mapping)")
        return  # Không add vào index hoặc mapping nếu ảnh lỗi!

    # Nếu cả text & image đều ok, add đồng thời vào index + mapping
    index_text.add(np.array([embedding_text], dtype='float32'))
    index_image.add(np.array([mean_vec], dtype='float32'))
    faiss.write_index(index_text, 'faiss_index.bin')
    faiss.write_index(index_image, 'faiss_index_image.bin')
    mapping = read_mapping()
    mapping.append(str(doc['_id']))
    write_mapping(mapping)
    print(f"Thêm mới: {doc['_id']}")

def sync_deleted(oid):
    global index_text, index_image
    mapping = read_mapping()
    oid_str = str(oid)
    if oid_str in mapping:
        pos = mapping.index(oid_str)
        # Xóa vector tại pos trong FAISS (text)
        mask = np.ones(index_text.ntotal, dtype=bool)
        mask[pos] = False
        vectors_text = index_text.reconstruct_n(0, index_text.ntotal)
        new_vectors_text = vectors_text[mask]
        new_index_text = faiss.IndexFlatIP(DIM_TEXT)
        new_index_text.add(np.array(new_vectors_text, dtype='float32'))
        faiss.write_index(new_index_text, 'faiss_index.bin')
        index_text = new_index_text

        # Xóa vector tại pos trong FAISS (image)
        mask_img = np.ones(index_image.ntotal, dtype=bool)
        mask_img[pos] = False
        vectors_image = index_image.reconstruct_n(0, index_image.ntotal)
        new_vectors_image = vectors_image[mask_img]
        new_index_image = faiss.IndexFlatIP(DIM_IMAGE)
        new_index_image.add(np.array(new_vectors_image, dtype='float32'))
        faiss.write_index(new_index_image, 'faiss_index_image.bin')
        index_image = new_index_image

        # Xóa _id khỏi mapping
        mapping.pop(pos)
        write_mapping(mapping)
        print(f"Xóa: {oid}")
    else:
        print(f"Không tìm thấy _id {oid} trong mapping. Không xóa.")

def sync_updated(doc):
    global index_text, index_image
    mapping = read_mapping()
    oid_str = str(doc['_id'])
    if oid_str in mapping:
        pos = mapping.index(oid_str)
        # 1. Xoá cũ khỏi cả 2 index
        mask = np.ones(index_text.ntotal, dtype=bool)
        mask[pos] = False
        vectors_text = index_text.reconstruct_n(0, index_text.ntotal)
        new_vectors_text = vectors_text[mask]
        new_index_text = faiss.IndexFlatIP(DIM_TEXT)
        new_index_text.add(np.array(new_vectors_text, dtype='float32'))

        mask_img = np.ones(index_image.ntotal, dtype=bool)
        mask_img[pos] = False
        vectors_image = index_image.reconstruct_n(0, index_image.ntotal)
        new_vectors_image = vectors_image[mask_img]
        new_index_image = faiss.IndexFlatIP(DIM_IMAGE)
        new_index_image.add(np.array(new_vectors_image, dtype='float32'))

        mapping.pop(pos)

        # 2. Thử lại với logic giống insert (image thử lại 5 lần)
        description_concat = build_description_concat(doc)
        embedding_text = model_text.encode(['passage: ' + description_concat], normalize_embeddings=True)[0]
        image_urls = [img['url'] for img in doc.get('electronicImgs', []) if img.get('url')]
        mean_vec = try_get_image_embedding(image_urls, swin_model, image_processor, max_retry=5)
        if mean_vec is None:
            # Không add lại vào mapping/index nếu ảnh vẫn lỗi
            write_mapping(mapping)
            faiss.write_index(new_index_text, 'faiss_index.bin')
            faiss.write_index(new_index_image, 'faiss_index_image.bin')
            index_text = new_index_text
            index_image = new_index_image
            print(f"⚠️ Cập nhật: {doc['_id']} nhưng ảnh lỗi, đã xoá khỏi index/mapping")
            return

        # Nếu cả text & image ok, add lại đồng thời
        new_index_text.add(np.array([embedding_text], dtype='float32'))
        new_index_image.add(np.array([mean_vec], dtype='float32'))
        mapping.append(oid_str)
        write_mapping(mapping)
        faiss.write_index(new_index_text, 'faiss_index.bin')
        faiss.write_index(new_index_image, 'faiss_index_image.bin')
        index_text = new_index_text
        index_image = new_index_image
        print(f"Cập nhật: {doc['_id']}")
    else:
        print(f"Không tìm thấy _id {doc['_id']} trong mapping để cập nhật.")

def main():
    print("⏳ Worker FAISS-MongoDB Sync (Text+Image, retry image 5x) đang chạy...")
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
