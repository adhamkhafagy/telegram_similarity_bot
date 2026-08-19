"""
بوت تليجرام للبحث عن التصاميم المشابهة بالصورة وإضافة تصاميم جديدة.

طريقة الاستخدام:
- ابعت صورة عادية -> البوت يدورلك على أقرب تصاميم مشابهة في الأرشيف
- لو التصميم جديد فعلاً وعايز تضيفه -> ابعت /add بعد الصورة مباشرة
"""
import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters

from embeddings import get_image_embedding
from vector_store import add_design, search_similar, mark_reused, get_design_file_id, get_top_reused

# بيقرأ ملف .env تلقائيًا ويحمّل المتغيرات اللي فيه (بديل عن export اليدوي)
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# نسبة التشابه اللي لو أعلى منها، بنعتبر التصميم موجود بالفعل في الأرشيف
DUPLICATE_THRESHOLD = 90.0

# تخزين مؤقت (في الميموري) لآخر embedding و file_id بعتهم كل مستخدم، عشان لو استخدم /add بعدها
_last_embedding: dict[int, list[float]] = {}
_last_file_id: dict[int, str] = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "أهلاً! ابعتلي صورة أي تصميم وهدورلك على أقرب تصاميم مشابهة ليها في الأرشيف.\n"
        "لو التصميم جديد فعلاً، اكتب /add بعد ما تبعت الصورة عشان أضيفه للأرشيف."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    photo = update.message.photo[-1]  # أعلى دقة متاحة من الصورة
    file = await photo.get_file()
    image_bytes = await file.download_as_bytearray()

    searching_msg = await update.message.reply_text("بدور في الأرشيف...")

    embedding = get_image_embedding(bytes(image_bytes))
    _last_embedding[user_id] = embedding
    _last_file_id[user_id] = photo.file_id  # بنخزنه عشان نقدر نرجّع الصورة الأصلية بعدين

    matches = search_similar(embedding, top_k=5)

    if not matches or matches[0]["similarity"] < 50:
        await searching_msg.edit_text(
            "معلقتش على حاجة مشابهة في الأرشيف.\n"
            "لو التصميم ده جديد، ابعت /add عشان أضيفه."
        )
        return

    reply_lines = ["أقرب التصاميم المشابهة:\n"]
    for m in matches:
        reply_lines.append(f"• كود {m['id']} — تشابه {m['similarity']}%")
    reply_lines.append("\nلمشاهدة صورة أي كود منهم، ابعتي /get متبوع بالكود، مثلاً:\n/get " + matches[0]["id"])

    if matches[0]["similarity"] >= DUPLICATE_THRESHOLD:
        reply_lines.append(f"\nالتصميم ده يبدو إنه موجود بالفعل (كود {matches[0]['id']}).")
        mark_reused(matches[0]["id"])
    else:
        reply_lines.append("\nلو مفيش تصميم مطابق فعلاً، ابعت /add لإضافته كتصميم جديد.")

    await searching_msg.edit_text("\n".join(reply_lines))


async def add_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    embedding = _last_embedding.get(user_id)
    file_id = _last_file_id.get(user_id)

    if not embedding:
        await update.message.reply_text("ابعت الصورة الأول، وبعدين اكتب /add.")
        return

    design_id = add_design(embedding, telegram_file_id=file_id)
    await update.message.reply_text(f"تم إضافة التصميم للأرشيف بكود: {design_id}")
    _last_embedding.pop(user_id, None)
    _last_file_id.pop(user_id, None)


async def get_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    بيرجع صورة التصميم الأصلية عن طريق الكود بتاعه. الاستخدام: /get الكود
    """
    if not context.args:
        await update.message.reply_text("اكتبي الكود بعد الأمر، مثلاً:\n/get a1b2c3d4")
        return

    design_id = context.args[0]
    file_id = get_design_file_id(design_id)

    if not file_id:
        await update.message.reply_text(f"مفيش صورة محفوظة للكود ده ({design_id}).")
        return

    await update.message.reply_photo(photo=file_id, caption=f"كود {design_id}")


async def top_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    بيعرض أكتر 10 تصاميم استخدامًا في الأرشيف، مرتبة تنازليًا حسب عدد مرات التكرار.
    """
    top_designs = get_top_reused(limit=10)

    if not top_designs:
        await update.message.reply_text("الأرشيف لسه فاضي، مفيش تصاميم متضافة.")
        return

    reply_lines = ["أكتر 10 تصاميم استخدامًا:\n"]
    for rank, design in enumerate(top_designs, start=1):
        times = design["metadata"].get("times_reused", 0)
        reply_lines.append(f"{rank}. كود {design['id']} — اتكرر {times} مرة")

    await update.message.reply_text("\n".join(reply_lines))


def main():
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    # زيادة مهلة الاتصال (Timeout) عشان لو النت بطيء الشوية البوت مايقعش من أول مرة
    app = (
        Application.builder()
        .token(token)
        .connect_timeout(30)
        .read_timeout(30)
        .get_updates_connect_timeout(30)
        .get_updates_read_timeout(30)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("add", add_command))
    app.add_handler(CommandHandler("get", get_command))
    app.add_handler(CommandHandler("top", top_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("البوت شغال...")
    app.run_polling()


if __name__ == "__main__":
    main()