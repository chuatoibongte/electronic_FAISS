def build_description_concat(doc):
    name = doc.get('name', '')
    brand = doc.get('brandName', '')
    main_category = doc.get('mainCategory', '')
    categories = ', '.join(doc.get('categories', []))
    price = doc.get('price', '')
    rating = doc.get('rating', '')
    description = doc.get('description', '')

    specs = []
    for group in doc.get('specifications', []):
        group_name = group.get('name', '')
        for attr in group.get('attributes', []):
            key = attr.get('name', '')
            value = attr.get('value', '')
            specs.append(f"{group_name} - {key}: {value}")
    specs_text = " | ".join(specs)

    description_concat = (
        f"Tên: {name} | Hãng: {brand} | Loại chính: {main_category} | "
        f"Nhóm: {categories} | Giá: {price} | Đánh giá: {rating} | "
        f"Mô tả: {description} | Thông số: {specs_text}"
    )
    return description_concat


import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np

def get_mean_image_embedding(image_url_list, swin_model, image_processor, device="cpu"):
    vectors = []
    for idx, url in enumerate(image_url_list):
        try:
            print(f"  → [Ảnh {idx+1}/{len(image_url_list)}] Đang tải: {url}")
            response = requests.get(url, timeout=10)
            print("    - Status code:", response.status_code)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            print("    - PIL mở ảnh OK:", image.size)
            
            # Tiền xử lý cho Swin Transformer
            inputs = image_processor(image, return_tensors="pt")

            print("    - Preprocess shape:", inputs['pixel_values'].shape)
            
            with torch.no_grad():
                outputs = swin_model(**inputs)
                # Swin base patch4 window7 output là [B, 1024]
                embedding = outputs.pooler_output
                # Chuẩn hóa vector (giống logic cũ)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            vectors.append(embedding.cpu().numpy()[0])
            print("    - Vector embedding OK (10 số đầu):", vectors[-1][:10])
        except Exception as e:
            print(f"    ✗ Lỗi với ảnh {url}: {e}")
    print(f"==> Tổng số ảnh embedding thành công: {len(vectors)}/{len(image_url_list)}")
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


