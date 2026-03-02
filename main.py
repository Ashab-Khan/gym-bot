import asyncio
import os
import nest_asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Load API keys from .env file
load_dotenv()
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

nest_asyncio.apply()


# SETUP


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm        = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)


# CAFE MENU → VECTOR DB


cafe_menu = [
    {"item": "Grilled Chicken Sandwich", "calories": 450, "protein": 35, "carbs": 30, "fats": 12},
    {"item": "Paneer Salad Bowl",         "calories": 320, "protein": 28, "carbs": 15, "fats": 10},
    {"item": "Egg White Omelette",        "calories": 200, "protein": 24, "carbs": 2,  "fats": 8},
    {"item": "Banana Protein Smoothie",   "calories": 380, "protein": 30, "carbs": 45, "fats": 5},
    {"item": "Chocolate Brownie",         "calories": 520, "protein": 5,  "carbs": 70, "fats": 25},
    {"item": "Oats Bowl with Fruits",     "calories": 350, "protein": 12, "carbs": 60, "fats": 6},
    {"item": "Grilled Fish with Rice",    "calories": 480, "protein": 42, "carbs": 40, "fats": 10},
]

docs = []
for item in cafe_menu:
    text = f"{item['item']} - Calories:{item['calories']}, Protein:{item['protein']}g, Carbs:{item['carbs']}g, Fats:{item['fats']}g"
    docs.append(Document(page_content=text))

vector_db = FAISS.from_documents(docs, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
print("✅ Vector DB ready!")


# USER SESSIONS


user_sessions = {}


# AI RESPONSE FUNCTION


def get_ai_response(user_id, user_message):
    session      = user_sessions[user_id]
    user_context = session["context"]
    chat_history = session["history"]

    relevant_docs = retriever.invoke(user_message)
    menu_context  = "\n".join([doc.page_content for doc in relevant_docs])

    messages = [
        ("system", f"""You are a friendly gym nutrition expert chatbot.
{user_context}
Relevant cafe menu items:
{menu_context}
Always suggest from cafe menu. Be encouraging and practical.
Keep responses concise for mobile reading.""")
    ]

    for human, ai in chat_history:
        messages.append(("human", human))
        messages.append(("ai", ai))

    messages.append(("human", user_message))

    response = llm.invoke(messages)
    ai_reply = response.content
    chat_history.append((user_message, ai_reply))

    return ai_reply


# TELEGRAM HANDLERS


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_sessions[user_id] = {
        "context": "",
        "history": [],
        "step": "collect_name"
    }
    await update.message.reply_text(
        "🏋️ Welcome to Gym Nutrition Bot!\n\n"
        "I will create a personalized nutrition plan for you "
        "based on your gym cafe menu.\n\n"
        "Let's start! What is your name?"
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_sessions[user_id] = {
        "context": "",
        "history": [],
        "step": "collect_name"
    }
    await update.message.reply_text(
        "🔄 Profile reset! Let's start fresh.\n\n"
        "What is your name?"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    message = update.message.text

    if user_id not in user_sessions:
        await update.message.reply_text("Please type /start to begin! 🏋️")
        return

    session = user_sessions[user_id]
    step    = session.get("step")

    if step == "collect_name":
        session["name"] = message
        session["step"] = "collect_age"
        await update.message.reply_text(f"Nice to meet you {message}! 👋\n\nHow old are you?")

    elif step == "collect_age":
        session["age"] = message
        session["step"] = "collect_sex"
        await update.message.reply_text("What is your sex? (Male/Female)")

    elif step == "collect_sex":
        session["sex"] = message
        session["step"] = "collect_weight"
        await update.message.reply_text("What is your current weight in kg?")

    elif step == "collect_weight":
        session["weight"] = message
        session["step"]   = "collect_height"
        await update.message.reply_text("What is your height in cm?")

    elif step == "collect_height":
        session["height"] = message
        session["step"]   = "collect_goal"
        await update.message.reply_text(
            "What is your goal?\n\n"
            "1️⃣ Weight Loss\n"
            "2️⃣ Muscle Gain\n"
            "3️⃣ Maintenance\n\n"
            "Type your goal:"
        )

    elif step == "collect_goal":
        session["goal"] = message
        try:
            weight = float(session["weight"])
            height = float(session["height"])
            bmi    = round(weight / ((height/100) ** 2), 1)
        except:
            bmi = "N/A"

        session["bmi"]  = bmi
        session["step"] = "chat"
        session["context"] = f"""
Member Name: {session['name']}
Age: {session['age']}
Sex: {session['sex']}
Weight: {session['weight']}kg
Height: {session['height']}cm
BMI: {bmi}
Goal: {session['goal']}
"""
        await update.message.reply_text(
            f"✅ Perfect {session['name']}! Your profile is ready.\n\n"
            f"📊 Your BMI: {bmi}\n"
            f"🎯 Goal: {session['goal']}\n\n"
            f"Ask me anything about nutrition! 💪\n\n"
            f"Tip: Type /reset anytime to update your profile."
        )

    elif step == "chat":
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )
        response = get_ai_response(user_id, message)
        await update.message.reply_text(response)


# MAIN


async def main():
    print("🤖 Starting Gym Nutrition Telegram Bot...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("✅ Bot is running!")
    await app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    asyncio.run(main())

