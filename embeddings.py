"""
وحدة توليد الـ Embeddings للصور باستخدام موديل CLIP.
بتحول أي صورة لمتجه رقمي (Vector) بنستخدمه للمقارنة بين التصاميم.
"""
import os
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
import io

# بنضبط عدد الـ Threads يدويًا بناءً على المعالجات المتاحة فعليًا،
# عشان نستغل كل الموارد المتاحة على السيرفر بدل الإعداد الافتراضي اللي مش دايمًا مثالي
_num_threads = os.cpu_count() or 1
torch.set_num_threads(_num_threads)

# تحميل الموديل مرة واحدة بس عند تشغيل السيرفر، مش كل مرة بيوصل فيها طلب
_model = SentenceTransformer("clip-ViT-B-32", device="cpu")


def get_image_embedding(image_bytes: bytes) -> list[float]:
    """
    بياخد بايتس الصورة، ويرجع الـ Embedding بتاعها كـ list أرقام.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    embedding = _model.encode(image)
    return embedding.tolist()
