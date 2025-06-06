import clip
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load CLIP model và MongoDB
clip_model, preprocess = clip.load("ViT-B/32", device="cpu")

load_dotenv()
MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client['ElectronicMaster_DB']
collection = db['electronics']

from utils import get_mean_image_embedding  # Nếu bạn lưu hàm trên vào file khác, còn không thì để ở cùng file

docs = list(collection.find({}).limit(100))
if not docs:
    print("Không có sản phẩm nào!")
    exit()

valid_count = 0
for i, doc in enumerate(docs):
    print(f"\n================= DOC {i} - {doc.get('_id')} =================")
    image_urls = [img['url'] for img in doc.get('electronicImgs', []) if img.get('url')]
    if not image_urls:
        print("⚠️ Sản phẩm này không có url ảnh.")
        continue
    mean_vec = get_mean_image_embedding(image_urls, clip_model, preprocess)
    if mean_vec is None:
        print(f"❌ Không embedding được ảnh cho sản phẩm {doc.get('_id')}")
        continue
    print(f"✔️ Sản phẩm {doc.get('_id')} OK, mean vector (10 số đầu): {mean_vec[:10]}")
    valid_count += 1

print(f"\nTổng số sản phẩm embedding ảnh thành công: {valid_count}/{len(docs)}")
