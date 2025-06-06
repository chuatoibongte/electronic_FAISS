from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from mangum import Mangum
DIM = 768
MODEL_NAME = 'intfloat/multilingual-e5-base'

app = FastAPI()
handler= Mangum(app)
index = faiss.read_index('faiss_index.bin')
model = SentenceTransformer(MODEL_NAME)

# Đọc mapping index <-> _id
with open('faiss_id_map.txt', encoding='utf-8') as f:
    faiss_id_map = [line.strip() for line in f]

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post('/search_ids')
async def search_ids(body: SearchRequest):
    query_vec = model.encode(['query: ' + body.query], normalize_embeddings=True)
    D, I = index.search(np.array(query_vec, dtype='float32'), body.top_k)
    # Lấy _id từ mapping
    ids = [faiss_id_map[idx] for idx in I[0]]
    scores = [float(score) for score in D[0]]
    return {"ids": ids, "scores": scores}
