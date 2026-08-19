"""
وحدة توليد الـ Embeddings للصور باستخدام موديل CLIP.
بتحول أي صورة لمتجه رقمي (Vector) بنستخدمه للمقارنة بين التصاميم.
"""
from sentence_transformers import SentenceTransformer
from PIL import Image
import io

# تحميل الموديل مرة واحدة بس عند تشغيل السيرفر، مش كل مرة بيوصل فيها طلب
_model = SentenceTransformer("clip-ViT-B-32")


def get_image_embedding(image_bytes: bytes) -> list[float]:
    """
    بياخد بايتس الصورة، ويرجع الـ Embedding بتاعها كـ list أرقام.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    embedding = _model.encode(image)
    return embedding.tolist()
