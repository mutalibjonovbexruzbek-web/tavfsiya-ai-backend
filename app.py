import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)

# GitHub Pages saytingizdan keladigan so'rovlarga ruxsat
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "https://mutalibjonovbexruzbek-web.github.io"
            ]
        }
    }
)

SYSTEM_PROMPT = """
Siz "Tavsiya AI" nomli aqlli va samimiy yordamchi ekansiz.

Sizning vazifangiz:
- Foydalanuvchini diqqat bilan tinglash.
- Muammoni tushunish va tahlil qilish.
- Amaliy va tushunarli tavsiyalar berish.
- Kerak bo'lsa muammoni bosqichma-bosqich yechishga yordam berish.
- Javoblarni juda qisqa qilmaslik.
- Muhim joylarda punktlar va raqamlardan foydalanish.
- Tabiiy va samimiy tilda yozish.
- Foydalanuvchi qaysi tilda yozsa, imkon qadar shu tilda javob berish.
- O'zbek tilida yozilganda tabiiy o'zbek tilidan foydalanish.
- Zarur joylarda mos emoji ishlatish, lekin haddan tashqari ko'p emas.
- Foydalanuvchiga hukm qilmaslik yoki masxara qilmaslik.
- Javobni aniq, foydali va amaliy qilish.

Tavsiya AI shifokor, psixolog, yurist yoki boshqa mutaxassisning o'rnini bosmaydi.
Agar masala professional yordamni talab qilsa, ishonchli katta odam yoki tegishli mutaxassisga murojaat qilishni tavsiya qiling.

Agar foydalanuvchi o'ziga zarar yetkazish yoki xavf ostida qolish haqida gapirsa:
- vaziyatni jiddiy qabul qiling;
- xavfsiz joyga o'tishni va ishonchli katta odamga darhol aytishni tavsiya qiling;
- zarur bo'lsa mahalliy favqulodda xizmatlarga murojaat qilishni ayting;
- xavfli harakatlarning tafsilotlarini bermang.

Javoblar foydalanuvchiga yordam berishga qaratilgan bo'lsin.
"""

MODEL = os.environ.get(
    "GROQ_MODEL",
    "qwen/qwen3.6-27b"
)


@app.get("/")
def home():
    return jsonify({
        "name": "Tavsiya AI Backend",
        "status": "online"
    })


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "tavsiya-ai-backend"
    })


@app.post("/api/chat")
def chat():
    try:
        data = request.get_json(silent=True) or {}

        messages = data.get("messages", [])

        if not messages:
            return jsonify({
                "error": "Xabar yuborilmadi."
            }), 400

        # Juda katta tarix yuborib, API sarfini oshirmaslik uchun
        messages = messages[-12:]

        api_key = os.environ.get("GROQ_API_KEY")

        if not api_key:
            return jsonify({
                "error": "GROQ_API_KEY serverda sozlanmagan."
            }), 500

        client = Groq(
            api_key=api_key,
            default_headers={
                "Groq-Model-Version": "latest"
            }
        )

        clean_messages = []

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role not in ["user", "assistant"]:
                continue

            if not isinstance(content, str):
                continue

            if not content.strip():
                continue

            clean_messages.append({
                "role": role,
                "content": content[:6000]
            })

        if not clean_messages:
            return jsonify({
                "error": "Yaroqli xabar topilmadi."
            }), 400

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                *clean_messages
            ],
            temperature=0.7,
            max_completion_tokens=1500
        )

        reply = response.choices[0].message.content

        return jsonify({
            "reply": reply
        })

    except Exception as e:
        print("SERVER ERROR:", str(e))

        return jsonify({
            "error": "AI bilan bog'lanishda xatolik yuz berdi."
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
