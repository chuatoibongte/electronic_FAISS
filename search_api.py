from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import faiss
import numpy as np
import clip
import torch
from sentence_transformers import SentenceTransformer
from mangum import Mangum
from PIL import Image
import io

# ==== Cấu hình ====
DIM_TEXT = 768
MODEL_NAME_TEXT = 'intfloat/multilingual-e5-base'
DIM_IMAGE = 512
MODEL_NAME_IMAGE = "ViT-B/32"
MAPPING_FILE = 'faiss_id_map.txt'

app = FastAPI()
handler = Mangum(app)

# ==== Load models, index, mapping ====
index_text = faiss.read_index('faiss_index.bin')
index_img = faiss.read_index('faiss_index_image.bin')
model_text = SentenceTransformer(MODEL_NAME_TEXT)
clip_model, preprocess = clip.load(MODEL_NAME_IMAGE, device="cpu")

with open(MAPPING_FILE, encoding='utf-8') as f:
    faiss_id_map = [line.strip() for line in f]

# ==== Pydantic model ====
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResponse(BaseModel):
    ids: list
    scores: list

# ==== API Search text ====
@app.post('/search_ids', response_model=SearchResponse)
async def search_ids(body: SearchRequest):
    query_vec = model_text.encode(['query: ' + body.query], normalize_embeddings=True)
    D, I = index_text.search(np.array(query_vec, dtype='float32'), body.top_k)
    ids = [faiss_id_map[idx] for idx in I[0]]
    scores = [float(score) for score in D[0]]
    return {"ids": ids, "scores": scores}

# ==== API Search image ====
@app.post('/search_image', response_model=SearchResponse)
async def search_image(file: UploadFile = File(...), top_k: int = 5):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = clip_model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    image_vec = embedding.cpu().numpy().astype('float32')
    D, I = index_img.search(image_vec, top_k)
    ids = [faiss_id_map[idx] for idx in I[0]]
    scores = [float(score) for score in D[0]]
    return {"ids": ids, "scores": scores}
