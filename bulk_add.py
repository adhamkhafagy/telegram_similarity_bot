"""
سكريبت لإضافة عدد كبير من التصاميم دفعة واحدة من مجلد على الجهاز، بدل ما تبعتي كل صورة يدويًا للبوت.

طريقة الاستخدام:
    python bulk_add.py "C:\\path\\to\\designs_folder" 123456789

- الباراميتر الأول: مسار المجلد اللي فيه صور التصاميم (jpg, jpeg, png).
- الباراميتر الثاني: الـ Chat ID بتاعك على تليجرام (جيبيه من @userinfobot).

السكريبت بيبعت كل صورة لنفسك على تليجرام عشان ياخد منها file_id (مطلوب لأمر /get بعدين)،
وبعدين يضيفها للأرشيف بكود جديد تلقائيًا. في تقرير نهائي بكل الأكواد اللي اتضافت.
"""
import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from telegram import Bot

from embeddings import get_image_embedding
from vector_store import add_design

load_dotenv()

# صيغ الصور المدعومة
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# فاصل زمني بسيط بين كل صورة والتانية عشان مانضربش حد الإرسال بتاع تليجرام
DELAY_BETWEEN_UPLOADS_SECONDS = 1.0


async def bulk_add(folder_path: str, chat_id: str):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    bot = Bot(token=token)

    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"المجلد ده مش موجود: {folder_path}")
        return

    image_files = sorted(
        f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        print("مفيش صور بصيغة jpg/jpeg/png في المجلد ده.")
        return

    print(f"لقيت {len(image_files)} صورة. هبدأ الإضافة...\n")

    added = []
    failed = []

    for i, image_path in enumerate(image_files, start=1):
        try:
            # نبعت الصورة لنفسنا على تليجرام عشان ناخد منها file_id
            with open(image_path, "rb") as f:
                message = await bot.send_photo(chat_id=chat_id, photo=f)

            file_id = message.photo[-1].file_id

            # نولّد الـ embedding من نفس بايتس الصورة المحلية
            image_bytes = image_path.read_bytes()
            embedding = get_image_embedding(image_bytes)

            design_id = add_design(embedding, source="bulk_import", telegram_file_id=file_id)

            added.append((image_path.name, design_id))
            print(f"[{i}/{len(image_files)}] {image_path.name} -> كود {design_id}")

        except Exception as e:
            failed.append((image_path.name, str(e)))
            print(f"[{i}/{len(image_files)}] فشل: {image_path.name} — {e}")

        # هدنة بسيطة بين كل صورة والتانية
        await asyncio.sleep(DELAY_BETWEEN_UPLOADS_SECONDS)

    print("\n===== التقرير النهائي =====")
    print(f"تم إضافة {len(added)} تصميم بنجاح.")
    if failed:
        print(f"فشل {len(failed)} تصميم:")
        for name, error in failed:
            print(f"  - {name}: {error}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('الاستخدام: python bulk_add.py "مسار_المجلد" الـ_CHAT_ID')
        sys.exit(1)

    folder_arg = sys.argv[1]
    chat_id_arg = sys.argv[2]

    asyncio.run(bulk_add(folder_arg, chat_id_arg))
