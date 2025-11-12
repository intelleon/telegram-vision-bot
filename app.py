import os
import base64
import traceback
from io import BytesIO
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from openai import OpenAI

# --- Configuration ---
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

# Default model: best available for image understanding
DEFAULT_MODEL = "o4-mini"
ALLOWED_MODELS = {"o4-mini", "gpt-4o", "gpt-4o-mini"}

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Prompt ---
INSTRUCTION = (
    "Describe the image in ≤100 words. State: 1) what it shows (main subject/action), "
    "2) where (landmark/city/country if inferable), 3) when (time of day/season/era if inferable). "
    "Use visible text, signage, architecture, plates, vegetation, clothing, and shadows as clues. "
    "If any item isn’t inferable, say “uncertain”. Prefer concise but complete descriptions."
)

# --- Utilities ---
def enforce_word_limit(text: str, limit: int = 100) -> str:
    words = text.strip().split()
    return " ".join(words[:limit]) + ("…" if len(words) > limit else "")

def current_model(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.application.bot_data.get("model", DEFAULT_MODEL)

def set_model(context: ContextTypes.DEFAULT_TYPE, model: str) -> None:
    context.application.bot_data["model"] = model

def is_reasoning_model(model: str) -> bool:
    return model.startswith("o")

SUPPORTED_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif"}

def validate_image(image_bytes: bytes, mime_type: str) -> tuple[bool, str]:
    if not image_bytes or len(image_bytes) < 256:
        return False, "The image appears empty or too small."
    if len(image_bytes) > 15 * 1024 * 1024:  # 15 MB limit
        return False, "The image is too large (>15 MB). Please send a smaller image."
    if mime_type.lower() in {"image/heic", "image/heif"}:
        return False, "HEIC/HEIF isn’t supported. Please resend as JPEG or PNG."
    if mime_type.lower() not in SUPPORTED_MIME:
        return False, f"Unsupported format ({mime_type}). Please resend as JPEG or PNG."
    return True, ""

# --- Telegram file helpers ---
async def _download_image_bytes_and_mime(update: Update) -> tuple[bytes | None, str, str]:
    """
    Returns (image_bytes, mime_type, source_type)
    source_type is 'photo' or 'document'
    """
    msg = update.message
    if msg is None:
        return None, "", ""

    # Photo (compressed JPEG)
    if msg.photo:
        f = await msg.photo[-1].get_file()
        buf = BytesIO()
        await f.download_to_memory(out=buf)
        return buf.getvalue(), "image/jpeg", "photo"

    # Image sent as document (original quality)
    if msg.document and msg.document.mime_type and msg.document.mime_type.startswith("image/"):
        f = await msg.document.get_file()
        buf = BytesIO()
        await f.download_to_memory(out=buf)
        return buf.getvalue(), msg.document.mime_type, "document"

    return None, "", ""

async def analyze_image(image_bytes: bytes, mime_type: str, model: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = mime_type if (mime_type and "/" in mime_type) else "image/jpeg"
    data_url = f"data:{mime};base64,{b64}"

    def build_kwargs_for(m: str):
        # o4-mini uses max_completion_tokens and disallows temperature
        if m == "o4-mini":
            return dict(
                model=m,
                max_completion_tokens=250,
                messages=[
                    {"role": "system", "content": "You are a concise, high-precision vision analyst. Use ≤100 words."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": INSTRUCTION},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
            )
        # gpt-4o and others use max_tokens and allow temperature
        return dict(
            model=m,
            temperature=0.1,
            max_tokens=250,
            messages=[
                {"role": "system", "content": "You are a concise, high-precision vision analyst. Use ≤100 words."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": INSTRUCTION},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )

    # 1) Try the selected model (o4-mini by default)
    try:
        resp = client.chat.completions.create(**build_kwargs_for(model))
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return enforce_word_limit(text, 100)
    except Exception:
        print("=== OpenAI API ERROR (primary) ===")
        traceback.print_exc()

    # 2) Fallback to gpt-4o if empty or failed
    try:
        resp2 = client.chat.completions.create(**build_kwargs_for("gpt-4o"))
        text2 = (resp2.choices[0].message.content or "").strip()
        if text2:
            return enforce_word_limit(text2, 100)
    except Exception:
        print("=== OpenAI API ERROR (fallback) ===")
        traceback.print_exc()

    # 3) Last resort: return a short, explicit message (non-empty)
    return "uncertain: I couldn’t extract a description from this image. Please resend as a photo or as a JPEG/PNG, or try /mode gpt-4o."

# --- Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Send me a photo (ideally as a *document* to preserve quality). "
        "I’ll describe what it shows, where, and when — in ≤100 words.\n\n"
        f"Running model: {current_model(context)}. Commands: /mode, /status, /help"
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ Send a photo or an image file (document) and I’ll reply in ≤100 words.\n"
        "• /mode <o4-mini|gpt-4o|gpt-4o-mini> — switch accuracy/speed\n"
        "• /status — show current model"
    )

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"🧠 Current model: {current_model(context)}")

async def mode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args or []
    if not args:
        await update.message.reply_text(
            f"Usage: /mode <model>\nAllowed: {', '.join(sorted(ALLOWED_MODELS))}\nCurrent: {current_model(context)}"
        )
        return
    model = args[0].strip()
    if model not in ALLOWED_MODELS:
        await update.message.reply_text(
            f"❌ Unknown model '{model}'. Allowed: {', '.join(sorted(ALLOWED_MODELS))}"
        )
        return
    set_model(context, model)
    await update.message.reply_text(f"✅ Switched to {model}")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    image_bytes, mime_type, source = await _download_image_bytes_and_mime(update)

    if not image_bytes:
        await msg.reply_text("I couldn’t read the file. Please send a photo or an image file.")
        return

    ok, why = validate_image(image_bytes, mime_type)
    if not ok:
        await msg.reply_text(f"⚠️ {why}")
        if source == "document":
            await msg.reply_text("Tip: resend it as a *photo* to auto-convert to JPEG.")
        return

    # --- send temporary "spinner" message ---
    thinking_msg = await msg.reply_text("⏳ Analyzing image… please wait a few seconds.")

    model = current_model(context)
    try:
        reply_text = await analyze_image(image_bytes, mime_type, model)
    except Exception as e:
        msg_text = str(e).strip() or "Unknown error"
        reply_text = f"⚠️ Sorry, I couldn’t analyze that image. {msg_text}"

    # Ensure non-empty reply
    if not reply_text or not reply_text.strip():
        reply_text = "uncertain: I couldn’t extract a description. Please resend as a photo or as JPEG/PNG."

    # --- edit previous message instead of sending a new one ---
    try:
        await thinking_msg.edit_text(reply_text)
    except Exception:
        # If editing fails (e.g. message too old), send a new one
        await msg.reply_text(reply_text, reply_to_message_id=msg.message_id)

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    # Log full error but avoid crashing
    print("=== TELEGRAM HANDLER ERROR ===")
    traceback.print_exc()
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text("⚠️ Something went wrong while handling that. Please try again.")
    except Exception:
        pass


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📸 Please send me an image (ideally as a *document* to preserve quality). "
        "I’ll describe what it shows, where it might be, and when — in ≤100 words."
    )

# --- Main ---
def main():
    print("🚀 Bot starting…")
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.bot_data["model"] = DEFAULT_MODEL

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("mode", mode_cmd))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_error_handler(on_error)


    print(f"✅ Bot is running with model: {app.bot_data['model']}")
    app.run_polling(allowed_updates=["message"])

if __name__ == "__main__":
    main()
