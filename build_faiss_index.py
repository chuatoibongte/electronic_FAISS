import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import clip
import torch
from utils import build_description_concat, get_mean_image_embedding
from bson import ObjectId

# ========== CÀI ĐẶT ==========
DIM_TEXT = 768
MODEL_NAME_TEXT = 'intfloat/multilingual-e5-base'
MODEL_NAME_IMAGE = "ViT-B/32"
DIM_IMAGE = 512  # ViT-B/32

load_dotenv()
MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client['ElectronicMaster_DB']
collection = db['electronics']
docs = list(collection.find({}))

# ========== LOAD MODELS ==========
print("🔹 Đang load model...")
text_model = SentenceTransformer(MODEL_NAME_TEXT)
clip_model, preprocess = clip.load(MODEL_NAME_IMAGE, device="cpu")

# ========== BẮT ĐẦU BUILD ==========
mongo_ids = []
text_embeddings = []
image_embeddings = []
failed_ids = []

print("🔹 Duyệt từng sản phẩm (build lần 1)...")
for doc in docs:
    # 1. Lấy embedding image trước
    image_urls = [img['url'] for img in doc.get('electronicImgs', []) if img.get('url')]
    mean_vec = get_mean_image_embedding(image_urls, clip_model, preprocess)
    if mean_vec is None:
        print(f"⚠️ Bỏ qua doc {doc.get('_id')} (không có image hợp lệ, sẽ retry sau)")
        failed_ids.append(str(doc.get('_id')))
        continue

    # 2. Nếu có ảnh hợp lệ, lấy embedding text
    description_concat = build_description_concat(doc)
    text_emb = text_model.encode("passage: " + description_concat, normalize_embeddings=True)

    # 3. Lưu vào list
    mongo_ids.append(str(doc['_id']))
    image_embeddings.append(mean_vec)
    text_embeddings.append(text_emb)

print(f"✅ Tổng số sản phẩm index bình thường: {len(mongo_ids)}")
print(f"🔁 Số sản phẩm cần retry (ghi ra failed_image_ids.txt): {len(failed_ids)}")

# Ghi danh sách failed ids ra file
with open('failed_image_ids.txt', 'w', encoding='utf-8') as f:
    for _id in failed_ids:
        f.write(_id + '\n')

# ========== BẮT ĐẦU RETRY CỨU ẢNH CHO FAILED IDS ==========
retry_success = 0
for failed_id in failed_ids:
    doc = collection.find_one({'_id': ObjectId(failed_id)})
    if doc is None:
        print(f"❌ Không tìm thấy doc {failed_id}")
        continue

    image_urls = [img['url'] for img in doc.get('electronicImgs', []) if img.get('url')]
    mean_vec = None
    # Thử tối đa 5 lần
    for attempt in range(5):
        mean_vec = get_mean_image_embedding(image_urls, clip_model, preprocess)
        if mean_vec is not None:
            print(f"✅ Đã cứu thành công ảnh cho sản phẩm {failed_id} ở lần thử {attempt+1}!")
            break
        else:
            print(f"⏳ Thử lại lần {attempt+1} cho sản phẩm {failed_id}...")

    if mean_vec is not None:
        description_concat = build_description_concat(doc)
        text_emb = text_model.encode("passage: " + description_concat, normalize_embeddings=True)
        mongo_ids.append(str(doc['_id']))
        image_embeddings.append(mean_vec)
        text_embeddings.append(text_emb)
        retry_success += 1
    else:
        print(f"⛔️ Thử 5 lần vẫn không lấy được ảnh cho {failed_id}. Bỏ qua luôn!")

print(f"✅ Đã cứu thành công thêm {retry_success} sản phẩm có ảnh lỗi!")

# ========== BUILD FAISS INDEX & LƯU ==========
if text_embeddings and image_embeddings:
    text_embeddings = np.array(text_embeddings, dtype='float32')
    image_embeddings = np.array(image_embeddings, dtype='float32')

    index_text = faiss.IndexFlatIP(text_embeddings.shape[1])
    index_img = faiss.IndexFlatIP(image_embeddings.shape[1])
    index_text.add(text_embeddings)
    index_img.add(image_embeddings)

    faiss.write_index(index_text, 'faiss_index.bin')
    faiss.write_index(index_img, 'faiss_index_image.bin')

    with open('faiss_id_map.txt', 'w', encoding='utf-8') as f:
        for _id in mongo_ids:
            f.write(_id + '\n')

    print("🎉 Đã build xong index và mapping đồng bộ cho cả text và image!")
else:
    print("❌ Không đủ sản phẩm để build index (kiểm tra lại dữ liệu hoặc pipeline)")

# ========== KẾT THÚC ==========
print(f"Tổng số vector text: {len(text_embeddings)}")
print(f"Tổng số vector image: {len(image_embeddings)}")
print("✅ Pipeline đã hoàn tất.")
