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
