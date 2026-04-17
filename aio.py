
import re
import os
import json
import sys
import asyncio
import logging
import numpy as np
import subprocess
from datetime import datetime, timezone
import random
from os import getenv
from pathlib import Path
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


from settings import (
    OPENAI_API_KEY,
    conn,
    tel_token,
    get_connection,
    usernames,
    bmw,
    mamka,
    mamka_response
)

from tool_calls import (
    search_and_extract,
    read_prompt,
    reboot_pi,
    write_prompt,
    get_current_system,
    update_system,
    git_pull,
    system
)

from functions import (
    delete_embedding_from_db,
    generate_embedding,
    save_embedding,
    find_similar_messages

)

from dict import *

from tools import tools

system = system()
save_accuracy = 0.65
search_accuracy = 0.33
max_output_tokens = 1000
model_name = "gpt-4.1-mini"
chat_history = deque(maxlen=15)

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).parent
SYSTEM_PATH = BASE_DIR / "system.txt"
DEFAULT_SYSTEM_PATH = BASE_DIR / "default_system.txt"

dp = Dispatcher()
router = Router()
dp.include_router(router)

conn.autocommit = True
cursor = conn.cursor()

client = OpenAI(api_key=OPENAI_API_KEY)


async def call_function(name, args):
    if name == "search_and_extract":
        return await search_and_extract(**args)
    elif name == "reboot_pi":
        return await reboot_pi()
    elif name == "git_pull":
        return await git_pull()
    elif name == "update_system":
        new_prompt = args["new_prompt"]
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
        name = usernames.get(str(user_id), 'невідоме')

        messages = [
                {
                    "role": "system",
                    "content": system
                },
                *chat_history,
                {
                    "role": "user",
                    "content": "Переслане повідомлення: " + quoted_message,
                },
                # {
                #     "role": "user",
                #     "content": similar_info,
                # },
                {
                    "role": "user",
                    "content": "імя співрозмовника: " + name
                },
                {
                    "role": "user",
                    "content": f"{cleaned_message_text}"
                }
            ]

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
                "output": str(result)
            })

        if function_called:
            for output in tool_outputs:
                messages.append({
                    "role": "user",
                    "content": f"Результат функції {func_name}: {output['output']}"
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

    elif 'стас' in cleaned_text or 'лена' in cleaned_text or 'лєна' in cleaned_text:

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

            if similar_messages:
                similar_info = "\n".join([f"схожа інформація є у базі: {msg[0]} автор:{usernames.get(str(msg[2]), 'невідоме')} (схожість: {msg[1]:.2f})" for msg in similar_messages])
                logging.info(f"схожа інформація є у базі: {msg[0]} (схожість: {msg[1]:.2f})" for msg in similar_messages)

            else:
                similar_info = "Схожих повідомленнь немає"

            name = usernames.get(str(user_id), 'невідоме')
            original_name = usernames.get(str(quoted_user_id), 'невідоме')

            messages = [
                {
                    "role": "system",
                    "content": system
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
                {
                    "role": "user",
                    "content": similar_info
                },
            ]

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
                    "output": str(result)
                })

            if function_called:

                # func results as user messages
                for output in tool_outputs:
                    messages.append({
                        "role": "user",
                        "content": f"Результат функції {func_name}: {output['output']}"
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