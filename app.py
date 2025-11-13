import os
import base64
import traceback
import sqlite3
from pathlib import Path
from datetime import datetime
from io import BytesIO

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from openai import OpenAI

# =========================
# Configuration
# =========================

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# Database path (can override with BOT_DB_PATH env if you like)
DB_PATH = os.getenv("BOT_DB_PATH", "/data/bot.db")

DEFAULT_MODEL = "o4-mini"
ALLOWED_MODELS = {"o4-mini", "gpt-4o", "gpt-4o-mini"}

WORD_LIMIT = 200

client = OpenAI(api_key=OPENAI_API_KEY)

SUPPORTED_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif"}

# =========================
# Internal ChatGPT instruction (Sherlock mode)
# =========================

INSTRUCTION = (
    "You are Sherlock Holmes examining the provided image.\n\n"
    "Follow this structure:\n"
    "1) Observations: Brief bullet list of key visual details only.\n"
    "2) Deductions: Brief bullet list of logical conclusions based strictly on the observations.\n\n"
    f"Total length (both sections together) must not exceed {WORD_LIMIT} words. "
    "Observe people, objects, clothing, posture, lighting, weather, architecture, text, reflections, "
    "shadows, traces, and any subtle evidence. "
    "Make your reasoning calm, precise, and characteristic of Sherlock Holmes—analytical and insightful. "
    "Never claim certainty where evidence is weak: use 'likely', 'appears', or 'uncertain'. "
    "Do not invent facts that cannot be strongly inferred from the image."
)

# =========================
# DB helpers
# =========================

def init_db() -> None:
    """Create database and tables if they do not exist."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            last_seen TEXT
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT,
            model TEXT,
            ts TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def get_db_connection():
    return sqlite3.connect(DB_PATH)


def save_user(update: Update) -> None:
    """Upsert basic user info and last_seen timestamp."""
    if not update.effective_user:
        return
    u = update.effective_user
    now = datetime.utcnow().isoformat(timespec="seconds")
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO users (id, username, first_name, last_name, last_seen)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            username = excluded.username,
            first_name = excluded.first_name,
            last_name = excluded.last_name,
            last_seen = excluded.last_seen
        """,
        (u.id, u.username, u.first_name, u.last_name, now),
    )
    conn.commit()
    conn.close()


def save_stat(update: Update, action: str, model: str | None = None) -> None:
    """Record a lightweight usage event."""
    if not update.effective_user:
        return
    user_id = update.effective_user.id
    now = datetime.utcnow().isoformat(timespec="seconds")
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO stats (user_id, action, model, ts) VALUES (?, ?, ?, ?)",
        (user_id, action, model, now),
    )
    conn.commit()
    conn.close()


# =========================
# Utility functions
# =========================

def enforce_word_limit(text: str, limit: int = WORD_LIMIT) -> str:
    """
    Enforce a word limit without breaking sentences when possible.
    """
    text = (text or "").strip()
    if not text:
        return text

    words = text.split()
    if len(words) <= limit:
        return text

    partial = " ".join(words[:limit]).strip()

    # Prefer ending on ., !, ?
    for punct in [".", "!", "?"]:
        idx = partial.rfind(punct)
        if idx != -1 and idx > 50:
            return partial[: idx + 1].strip()

    # Next best: :, ;, —, –, ,
    for punct in [":", ";", "—", "–", ","]:
        idx = partial.rfind(punct)
        if idx != -1 and idx > 50:
            return partial[: idx + 1].strip()

    return partial


def current_model(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.application.bot_data.get("model", DEFAULT_MODEL)


def set_model(context: ContextTypes.DEFAULT_TYPE, model: str) -> None:
    context.application.bot_data["model"] = model


def validate_image(image_bytes: bytes, mime_type: str) -> tuple[bool, str]:
    if not image_bytes or len(image_bytes) < 256:
        return False, "The image appears empty or too small."
    if len(image_bytes) > 15 * 1024 * 1024:
        return False, "The image is too large (>15 MB). Please send a smaller image."
    if mime_type.lower() in {"image/heic", "image/heif"}:
        return False, "HEIC/HEIF is not supported. Please resend as JPEG or PNG."
    if mime_type.lower() not in SUPPORTED_MIME:
        return False, f"Unsupported format ({mime_type}). Please resend as JPEG or PNG."
    return True, ""


# =========================
# Telegram helpers
# =========================

async def _download_image_bytes_and_mime(update: Update) -> tuple[bytes | None, str, str]:
    """
    Returns (image_bytes, mime_type, source_type)
    source_type is 'photo' or 'document' or ''.
    """
    msg = update.message
    if msg is None:
        return None, "", ""

    # Photo
    if msg.photo:
        f = await msg.photo[-1].get_file()
        buf = BytesIO()
        await f.download_to_memory(out=buf)
        return buf.getvalue(), "image/jpeg", "photo"

    # Image as document
    if msg.document and msg.document.mime_type and msg.document.mime_type.startswith("image/"):
        f = await msg.document.get_file()
        buf = BytesIO()
        await f.download_to_memory(out=buf)
        return buf.getvalue(), msg.document.mime_type, "document"

    return None, "", ""


# =========================
# OpenAI interaction
# =========================

async def analyze_image(image_bytes: bytes, mime_type: str, model: str) -> str:
    """
    Analyze image with the selected model. If it fails or returns empty text,
    fall back to gpt-4o.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = mime_type if (mime_type and "/" in mime_type) else "image/jpeg"
    data_url = f"data:{mime};base64,{b64}"

    def build_kwargs_for(m: str):
        system_content = (
            "You are Sherlock Holmes, a concise, high-precision vision analyst.\n\n"
            f"{INSTRUCTION}"
        )

        if m == "o4-mini":
            return dict(
                model=m,
                max_completion_tokens=350,
                messages=[
                    {"role": "system", "content": system_content},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": INSTRUCTION},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
            )
        # gpt-4o & friends
        return dict(
            model=m,
            temperature=0.3,
            max_tokens=350,
            messages=[
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": INSTRUCTION},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )

    # 1) Try selected model
    try:
        resp = client.chat.completions.create(**build_kwargs_for(model))
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return enforce_word_limit(text, WORD_LIMIT)
    except Exception:
        print("=== OpenAI API ERROR (primary) ===")
        traceback.print_exc()

    # 2) Fallback to gpt-4o
    try:
        resp2 = client.chat.completions.create(**build_kwargs_for("gpt-4o"))
        text2 = (resp2.choices[0].message.content or "").strip()
        if text2:
            return enforce_word_limit(text2, WORD_LIMIT)
    except Exception:
        print("=== OpenAI API ERROR (fallback) ===")
        traceback.print_exc()

    return (
        "🤔 Curious. I am unable to draw reliable conclusions from this image. "
        "Perhaps a clearer photograph or different angle would yield better clues."
    )


async def analyze_image_gemini(image_bytes: bytes, mime_type: str) -> str:
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")

        img = {
            "mime_type": mime_type,
            "data": image_bytes,
        }

        prompt = (
            "Analyze the image exactly like Sherlock Holmes would.\n"
            "Provide two sections:\n"
            "1) Observations (bullet points)\n"
            "2) Deductions (bullet points)\n"
            f"Limit the entire response to {WORD_LIMIT} words.\n"
            "Be factual, precise, and avoid hallucinations.\n"
        )

        resp = model.generate_content([prompt, img])
        text = resp.text or ""
        return enforce_word_limit(text)

    except Exception as e:
        print("=== Gemini ERROR ===")
        print(e)
        return "⚠️ Gemini analysis failed. (Invalid image or API error.)"

# =========================
# Command handlers
# =========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_user(update)
    save_stat(update, "command_start", current_model(context))
    await update.message.reply_text(
        "🕵️‍♂️ Greetings. I am Sherlock Holmes. Send me a photograph, and I shall examine it with utmost precision. "
        "I will describe what is visible, identify notable clues, and offer logical deductions — all within "
        f"{WORD_LIMIT} words.\n\n"
        f"Current analytical engine: {current_model(context)}.\n"
        "Commands: /mode, /status, /help, /users, /stats, /top"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_user(update)
    save_stat(update, "command_help", current_model(context))
    await update.message.reply_text(
        "ℹ️ Simply send me a photograph or an image file. I shall inspect it as any competent detective would, "
        "describing observable details and inferring what the scene may reveal.\n\n"
        "Commands:\n"
        "/mode <o4-mini|gpt-4o|gpt-4o-mini> — select the analytical engine\n"
        "/status — show which engine I am presently using\n"
        "/users — summary of known visitors\n"
        "/stats — overall usage figures\n"
        "/top — users with the most analyzed images"
    )


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_user(update)
    save_stat(update, "command_status", current_model(context))
    await update.message.reply_text(
        f"🔍 Analytical engine currently in use: {current_model(context)}"
    )


async def mode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_user(update)
    args = context.args or []
    if not args:
        await update.message.reply_text(
            "To change the analytical engine, use:\n"
            "/mode <model>\n\n"
            f"Acceptable models: {', '.join(sorted(ALLOWED_MODELS))}\n"
            f"Currently in use: {current_model(context)}"
        )
        return

    model = args[0].strip()
    if model not in ALLOWED_MODELS:
        await update.message.reply_text(
            f"❌ That engine does not exist, my friend. Acceptable choices are: "
            f"{', '.join(sorted(ALLOWED_MODELS))}."
        )
        return

    set_model(context, model)
    save_stat(update, "command_mode_change", model)
    await update.message.reply_text(
        f"✔️ Very good. I shall henceforth conduct my analysis using: {model}."
    )


# ----- Tracking / admin commands -----

async def users_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_user(update)
    save_stat(update, "command_users", current_model(context))

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0] or 0

    c.execute(
        "SELECT id, username, first_name, last_seen "
        "FROM users ORDER BY last_seen DESC LIMIT 10"
    )
    rows = c.fetchall()
    conn.close()

    lines = [f"👥 Known visitors: {total_users}"]
    if rows:
        lines.append("\nLast 10 seen:")
        for uid, username, first_name, last_seen in rows:
            label = username or first_name or str(uid)
            lines.append(f"- {label} (id {uid}, last_seen {last_seen})")

    await update.message.reply_text("\n".join(lines))


async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_user(update)
    save_stat(update, "command_stats", current_model(context))

    conn = get_db_connection()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM stats")
    total_events = c.fetchone()[0] or 0

    c.execute("SELECT COUNT(DISTINCT user_id) FROM stats")
    active_users = c.fetchone()[0] or 0

    c.execute("SELECT action, COUNT(*) FROM stats GROUP BY action")
    action_rows = c.fetchall()

    conn.close()

    lines = [
        "📊 Usage summary:",
        f"- Total recorded events: {total_events}",
        f"- Distinct users with events: {active_users}",
    ]
    if action_rows:
        lines.append("\nBy action:")
        for action, count in action_rows:
            lines.append(f"- {action}: {count}")

    await update.message.reply_text("\n".join(lines))


async def top_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_user(update)
    save_stat(update, "command_top", current_model(context))

    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """
        SELECT u.id, COALESCE(u.username, u.first_name, u.id), COUNT(s.id) as cnt
        FROM stats s
        JOIN users u ON u.id = s.user_id
        WHERE s.action = 'image_analyzed'
        GROUP BY u.id
        ORDER BY cnt DESC
        LIMIT 10
        """
    )
    rows = c.fetchall()
    conn.close()

    if not rows:
        await update.message.reply_text(
            "📈 No image analyses recorded yet. Once a few cases pass my desk, "
            "I shall reveal the most active patrons."
        )
        return

    lines = ["📈 Top image patrons (by analyzed images):"]
    for uid, label, cnt in rows:
        lines.append(f"- {label} (id {uid}): {cnt} images")

    await update.message.reply_text("\n".join(lines))


# =========================
# Message handlers
# =========================

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_user(update)
    model = current_model(context)

    image_bytes, mime_type, source = await _download_image_bytes_and_mime(update)
    msg = update.message

    if not image_bytes:
        save_stat(update, "image_missing", model)
        await msg.reply_text(
            "🧐 I perceive no readable image in your message. "
            "Please send a photograph or an image file."
        )
        return

    ok, why = validate_image(image_bytes, mime_type)
    if not ok:
        save_stat(update, "image_invalid", model)
        await msg.reply_text(f"⚠️ Unfortunately, I cannot examine this file: {why}")
        if source == "document":
            await msg.reply_text(
                "Tip: resend it as a *photo* so it is automatically converted to JPEG."
            )
        return

    thinking_msg = await msg.reply_text(
        "⏳ One moment. I am studying the image with a detective’s eye. "
        "Clues take but a moment to reveal themselves…"
    )

    try:

        chatgpt_answer = await analyze_image(image_bytes, mime_type, model)
        gemini_answer = await analyze_image_gemini(image_bytes, mime_type)

        reply_text = (
            "🧠 *ChatGPT (Sherlock Holmes)*:\n"
            f"{chatgpt_answer}\n\n"
            "🔮 *Gemini*:\n"
            f"{gemini_answer}"
        )

        save_stat(update, "image_analyzed", model)
    except Exception:
        print("=== analyze_image ERROR ===")
        traceback.print_exc()
        save_stat(update, "image_error", model)
        reply_text = (
            "⚠️ A complication has arisen during analysis. Even the sharpest detective "
            "encounters the occasional obstacle. Please try resending the image."
        )

    if not reply_text or not reply_text.strip():
        reply_text = (
            "🤔 Curious. I am unable to draw reliable conclusions from this image. "
            "Perhaps a clearer photograph would yield better clues."
        )

    try:
        await thinking_msg.edit_text(reply_text, parse_mode="Markdown")
    except Exception:
        await msg.reply_text(reply_text, reply_to_message_id=msg.message_id)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_user(update)
    save_stat(update, "text_message", current_model(context))
    await update.message.reply_text(
        "📸 Kindly provide an image, my dear friend. Words alone offer few clues. "
        "Send a photo, and I shall examine it with a detective’s precision."
    )


# =========================
# Global error handler
# =========================

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    print("=== TELEGRAM HANDLER ERROR ===")
    traceback.print_exc()
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(
                "⚠️ Something unexpected has occurred. Even Holmes is not immune to mishap. "
                "Please try again."
            )
    except Exception:
        pass


# =========================
# Main entry point
# =========================

def main():
    print("🚀 Bot starting…")

    # Ensure DB is ready
    init_db()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.bot_data["model"] = DEFAULT_MODEL

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("mode", mode_cmd))
    app.add_handler(CommandHandler("users", users_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("top", top_cmd))

    # Messages
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Errors
    app.add_error_handler(on_error)

    print(f"✅ Bot is running with model: {app.bot_data['model']}")
    app.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
