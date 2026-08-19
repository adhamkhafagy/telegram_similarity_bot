"""
وحدة التعامل مع ChromaDB لتخزين والبحث عن التصاميم بالتشابه البصري.
"""
import chromadb
import uuid
from datetime import datetime

# تخزين دائم على القرص (Persistent) عشان الأرشيف يفضل موجود بعد أي إعادة تشغيل
_client = chromadb.PersistentClient(path="./chroma_data")
_collection = _client.get_or_create_collection(
    name="designs",
    metadata={"hnsw:space": "cosine"},  # Cosine Similarity، أنسب مقياس لمقارنة الصور
)


def add_design(embedding: list[float], design_id: str | None = None, source: str = "telegram", telegram_file_id: str | None = None) -> str:
    """
    بيضيف تصميم جديد للأرشيف، وبيرجع الكود (ID) بتاعه.
    بيخزن file_id بتاع تليجرام لو موجود، عشان نقدر نرجّع الصورة الأصلية بعدين بالكود.
    """
    design_id = design_id or str(uuid.uuid4())[:8]
    _collection.add(
        ids=[design_id],
        embeddings=[embedding],
        metadatas=[{
            "added_at": datetime.utcnow().isoformat(),
            "source": source,
            "times_reused": 0,
            "telegram_file_id": telegram_file_id or "",
        }],
    )
    return design_id


def search_similar(embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    بيدور على أقرب top_k تصاميم مشابهة، وبيرجعهم مع نسبة التشابه لكل واحد.
    """
    if _collection.count() == 0:
        return []

    results = _collection.query(query_embeddings=[embedding], n_results=min(top_k, _collection.count()))

    matches = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity_pct = round((1 - distance) * 100, 1)  # تحويل المسافة لنسبة تشابه مفهومة
        matches.append({
            "id": results["ids"][0][i],
            "similarity": similarity_pct,
            "metadata": results["metadatas"][0][i],
        })
    return matches


def mark_reused(design_id: str):
    """
    بيزود عداد مرات إعادة استخدام التصميم، مفيد لمعرفة أكتر التصاميم تكرارًا شهريًا.
    """
    existing = _collection.get(ids=[design_id])
    if existing["ids"]:
        meta = existing["metadatas"][0]
        meta["times_reused"] = meta.get("times_reused", 0) + 1
        _collection.update(ids=[design_id], metadatas=[meta])


def get_design_file_id(design_id: str) -> str | None:
    """
    بيرجع telegram_file_id بتاع تصميم معين عن طريق كوده، عشان نقدر نبعت الصورة الأصلية تاني.
    """
    existing = _collection.get(ids=[design_id])
    if not existing["ids"]:
        return None
    file_id = existing["metadatas"][0].get("telegram_file_id")
    return file_id or None


def get_top_reused(limit: int = 10) -> list[dict]:
    """
    بيرجع أكتر التصاميم استخدامًا (times_reused) بترتيب تنازلي.
    """
    if _collection.count() == 0:
        return []

    all_designs = _collection.get()  # بيرجع كل التصاميم بميتاداتاهم

    designs = [
        {"id": all_designs["ids"][i], "metadata": all_designs["metadatas"][i]}
        for i in range(len(all_designs["ids"]))
    ]

    designs.sort(key=lambda d: d["metadata"].get("times_reused", 0), reverse=True)
    return designs[:limit]