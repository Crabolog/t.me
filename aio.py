
import re
import os
import json
import sys
import asyncio
import logging
import aiohttp
import numpy as np
import subprocess
import random
import openai
from openai import OpenAI
from collections import deque
from aiogram import Bot, Dispatcher, html, F, types, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from os import getenv
from pathlib import Path

from settings import (
    OPENAI_API_KEY,
    conn,
    tel_token,
    get_connection,
    usernames,
    bmw,
    mamka,
    mamka_response,
    question_marks,
    random_keyword,
    random_response,
    bingo_trigger,
    bingo_list,
    roll
)

from tool_calls import (
    search_and_extract,
    read_prompt,
    reboot_pi,
    write_prompt,
    get_current_system,
    update_system,
    git_pull,
)

from functions import (
    delete_embedding_from_db,
    generate_embedding,
    save_embedding,
    find_similar_messages

)

from dict import *

from tools import tools

# OpenAI embeddings use cosine similarity; no single official cutoff exists.
# Keep duplicate detection somewhat loose so the bot stores useful memories more often.
save_accuracy = 0.72
search_accuracy = 0.38
max_output_tokens = 1000
model_name = "gpt-5.4-nano-2026-03-17"
chat_history = deque(maxlen=15)

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).parent
SYSTEM_PATH = BASE_DIR / "system.txt"
DEFAULT_SYSTEM_PATH = BASE_DIR / "default_system.txt"

dp = Dispatcher()
router = Router()
dp.include_router(router)

cursor = None

client = OpenAI(api_key=OPENAI_API_KEY)


def build_memory_hint(similar_messages):
    if not similar_messages:
        return None

    for saved_text, similarity, user_id in similar_messages:
        short_text = re.sub(r"\s+", " ", saved_text or "").strip()
        author = usernames.get(str(user_id), "невідоме")
        logging.info(
            "Similar memory candidate: %s | similarity=%.2f | author=%s",
            short_text,
            similarity,
            author,
        )

    summaries = []
    for saved_text, similarity, user_id in similar_messages[:4]:
        short_text = re.sub(r"\s+", " ", saved_text or "").strip()
        if len(short_text) > 140:
            short_text = short_text[:137] + "..."
        author = usernames.get(str(user_id), "невідоме")
        summaries.append(f"{short_text} (схожість {similarity:.2f}, автор {author})")

    if not summaries:
        return None

    compact_summary = "; ".join(summaries)
    if len(compact_summary) > 320:
        compact_summary = compact_summary[:317] + "..."

    return (
        "Контекст із пам'яті (не основа для відповіді): "
        "є кілька схожих спогадів про попередні теми, збережених у базі; "
        f"підсумок: {compact_summary}"
    )


def should_save_embedding(text: str) -> bool:
    if not text:
        return False

    cleaned = re.sub(r"\s+", " ", text.strip())
    if len(cleaned) < 20 or len(cleaned) > 800:
        return False

    lowered = cleaned.lower()
    if any(marker in lowered for marker in ["http://", "https://", "@", "/start", "/default", "/delete"]):
        return False

    if any(greeting in lowered for greeting in ["привіт", "привет", "hello", "hi", "дякую", "спасиб", "thanks", "ok", "ок", "добрий день", "добрый день", "пока", "bye"]):
        return False

    if "?" in cleaned or "!" in cleaned:
        return False

    return True


async def call_function(name, args):
    args = args or {}
    if name == "search_and_extract":
        return await search_and_extract(**args)
    elif name == "reboot_pi":
        return await reboot_pi()
    elif name == "git_pull":
        return await git_pull()
    elif name == "update_system":
        new_prompt = args.get("new_prompt", "")
        return await update_system(new_prompt)
    # elif name == "generate_image":
    #     prompt = args.get("prompt", "")
    #     try:
    #         response = openai.images.generate(
    #             model="dall-e-2",
    #             prompt=prompt,
    #             size="256x256",
    #             n=1
    #         )
    #         image_url = response.data[0].url
    #         await message.answer_photo(photo=image_url, caption=f"Зображення за запитом: {prompt}")
    #         result = "123"  # чтобы не добавлять текст в messages
    #     except Exception as e:
    #         result = f"Помилка генерації зображення: {e}"


@dp.message(Command("default"))
async def sys_default(message: Message):
    default = read_prompt(DEFAULT_SYSTEM_PATH)
    write_prompt(SYSTEM_PATH, default)
    await message.reply("System оновлено до дефолтного значення")

    # Ensure the next request uses the reset prompt immediately.
    if hasattr(message, "bot"):
        await message.bot.send_chat_action(message.chat.id, "typing")


@dp.message(Command("delete"))
async def delete_embedding_handler(message: Message):
    text = message.text.strip()
    args = text.split(maxsplit=1)

    if len(args) > 1:
        embedding_text = args[1]
        deleted = await delete_embedding_from_db(embedding_text)

        if deleted:
            await message.reply(f"Дані з текстом '{embedding_text}' було видалено.")
        else:
            await message.reply(f"Даних для тексту '{embedding_text}' не знайдено в базі.")
    else:
        await message.reply("Будь ласка, вкажіть текст для видалення. Формат: /delete <текст>")

# -----------------bingo--------------------
@dp.message(F.text.in_(bingo_trigger))
async def bingo_command(message: Message):
    try:
        text = random.choice(bingo_list)
    except IndexError:
        text = 'Спробуй ще разок'
    await message.answer(text=text,reply_markup=None)


# ----------------roll--------------------
@dp.message(F.text.in_(roll))
async def roll_command(message: Message):
    text = random.randint(0, 100)
    await message.answer(text=f"{html.bold(message.from_user.full_name)} зролив {text}",reply_markup=None)

# ----------------bitcoin------------------
@dp.message(F.text.in_({'BTC', 'btc', '/btc', '/btc@ZradaLevelsBot', 'btc@ZradaLevelsBot'}))
async def btc_command(message: Message, bot: Bot):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT',timeout=15) as resp:
                data =  await resp.json()
                symbol = data['symbol']
                price = float(data['price'])
                price = "{:.2f}".format(price)
    except Exception:
        price = 'Спробуй ще разок'
    user_id = message.from_user.id
    bot_user = await bot.get_me()
    bot_id = bot_user.id
    response_text = (
        f"Ціна {symbol}: {price} USDT\n"
        )

    await message.answer(text=response_text, reply_markup=None)


@dp.message(lambda message: message.reply_to_message and message.reply_to_message.from_user.id == 6694398809)
async def handle_bot_reply(message: types.Message, bot: Bot):

    user_id = message.from_user.id if message.from_user.id else 0
    bot_user = await bot.get_me()
    bot_id = bot_user.id
    bot_name = usernames.get(str(bot_id), 'невідоме')
    time = datetime.now(timezone.utc).isoformat(sep=" ", timespec="seconds")

    cleaned_text = re.sub(
        r"[-()\"#/@;:<>{}`+=~|.!,]", "", message.text.lower()
    )

    cleaned_message_text = re.sub(
        r'^\s*стас[,\s]+', '', message.text, flags=re.IGNORECASE
    ).strip()

    quoted_message = message.reply_to_message.text if message.reply_to_message else message.text

    if not quoted_message and message.reply_to_message:
        if message.reply_to_message.caption:
            quoted_message = message.reply_to_message.caption
        else:
            quoted_message = "повідомлення без тексту"

    try:
        embedding = generate_embedding(cleaned_message_text)
        similar_messages = await find_similar_messages(embedding)
        memory_hint = build_memory_hint(similar_messages)

        if memory_hint:
            logging.info("Prepared memory hint for reply handler: %s", memory_hint)

        name = usernames.get(str(user_id), 'невідоме')

        messages = [
            {
                "role": "system",
                "content": get_current_system(model_name)
            },
            *chat_history,
            {
                "role": "user",
                "content": "Переслане повідомлення: " + quoted_message,
            },
            {
                "role": "user",
                "content": "імя співрозмовника: " + name
            },
            {
                "role": "user",
                "content": f"{cleaned_message_text}"
            }
        ]

        if memory_hint:
            messages.append({
                "role": "user",
                "content": memory_hint
            })

        response = client.responses.create(
            input=messages,
            model=model_name,
            tools=tools,
            # max_output_tokens=max_output_tokens
        )
        function_called = False
        tool_outputs = []
        for tool_call in response.output:
            if tool_call.type != "function_call":
                continue

            function_called = True
            func_name = tool_call.name
            args = json.loads(tool_call.arguments)
            result = await call_function(func_name, args)

            tool_outputs.append({
                "call_id": tool_call.call_id,
                "name": func_name,
                "output": str(result)
            })

        if function_called:
            for output in tool_outputs:
                messages.append({
                    "role": "user",
                    "content": f"Результат функції {output['name']}: {output['output']}"
                })
            response = client.responses.create(
                input=messages,
                model=model_name,
                tools=tools
                # max_output_tokens=max_output_tokens
            )

        reply = response.output_text
        await message.answer(reply, reply_markup=None, parse_mode=None)

        chat_history.append({
            "role": "user",
            "content": f"{cleaned_message_text}"
        })

        chat_history.append({
            "role": "assistant",
            "content": f"{reply}"
        })

    except Exception as e:
        logging.error(e)
        await message.answer(f"Ой вей: {e}")


@dp.message(F.text)
async def random_message(message: Message, bot: Bot):

    cleaned_text = re.sub(
        r"[-()\"#/@;:<>{}`+=~|.!,]", "", message.text.lower()
    )

    cleaned_message_text = re.sub(
        r'^\s*[,\s]+', '', message.text, flags=re.IGNORECASE
    ).strip()

    user_id = message.from_user.id if message.from_user.id else 0
    bot_user = await bot.get_me()
    bot_id = bot_user.id
    bot_name = usernames.get(str(bot_id), 'невідоме')
    name = usernames.get(str(user_id), 'невідоме')
    time = datetime.now(timezone.utc).isoformat(sep=" ", timespec="seconds")

    chat_history.append({
        "role": "user",
        "content": f"{cleaned_message_text}"
    })

    if any(keyword in cleaned_text for keyword in bmw):
        logging.info("bmw handler triggered.")
        await message.answer("Беха топ", reply_markup=None)

    elif any(keyword in cleaned_text for keyword in mamka):
        logging.info("mamka handler triggered.")
        await message.answer(random.choice(mamka_response))

    elif any(keyword in cleaned_text for keyword in random_keyword):
        logging.info("mamka handler triggered.")
        await message.answer(random.choice(random_response))

    elif 'стас' in cleaned_text or 'лена' in cleaned_text or 'лєна' in cleaned_text:

        if should_save_embedding(cleaned_message_text):
            embedding = generate_embedding(cleaned_message_text)
            await save_embedding(cleaned_message_text, embedding, user_id)

        quoted_message = (
            message.reply_to_message.text if message.reply_to_message and message.reply_to_message.text else "повідомлення без тексту"
        )

        quoted_userid = (
            message.reply_to_message.from_user.id if message.reply_to_message and message.reply_to_message.from_user else 0
        )

        quoted_user_id = quoted_userid if quoted_userid else 0

        try:

            embedding = generate_embedding(cleaned_message_text)
            similar_messages = await find_similar_messages(embedding)
            memory_hint = build_memory_hint(similar_messages)

            if memory_hint:
                logging.info("Prepared memory hint: %s", memory_hint)

            name = usernames.get(str(user_id), 'невідоме')
            original_name = usernames.get(str(quoted_user_id), 'невідоме')

            messages = [
                {
                    "role": "system",
                    "content": get_current_system(model_name)
                },
                *chat_history,
                {
                    "role": "user",
                    "content": "Переслане повідомлення: " + quoted_message,
                },
                {
                    "role": "user",
                    "content": "Автор пересланого повідомлення: " + original_name,
                },
                {
                    "role": "user",
                    "content": "імя співрозмовника: " + name
                },
                {
                    "role": "user",
                    "content": f"{cleaned_message_text}"
                },
            ]

            if memory_hint:
                messages.append({
                    "role": "user",
                    "content": memory_hint
                })

            response = client.responses.create(
                input=messages,
                model=model_name,
                tools=tools,
                # max_output_tokens=max_output_tokens
            )

            function_called = False
            tool_outputs = []

            for tool_call in response.output:
                if tool_call.type != "function_call":
                    continue

                function_called = True
                func_name = tool_call.name
                args = json.loads(tool_call.arguments)

                result = await call_function(func_name, args)

                # save for next request
                tool_outputs.append({
                    "call_id": tool_call.call_id,
                    "name": func_name,
                    "output": str(result)
                })

            if function_called:
                # func results as user messages
                for output in tool_outputs:
                    messages.append({
                        "role": "user",
                        "content": f"Результат функції {output['name']}: {output['output']}"
                    })

                response = client.responses.create(
                    input=messages,
                    model=model_name,
                    tools=tools
                    # max_output_tokens=max_output_tokens
                )

            reply = response.output_text

            await message.answer(reply, reply_markup=None, parse_mode=None)

            chat_history.append({
                "role": "assistant",
                "content": f"{reply}"
            })

        except Exception as e:
            logging.error(e)
            await message.answer(f"Ой вей: {e}")


async def main() -> None:
    bot = Bot(token=tel_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())