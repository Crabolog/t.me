import asyncio
import aiohttp
import logging
import re
import json
from collections import deque
import os
import sys
from pathlib import Path
import subprocess
from os import getenv
from settings import conn
from settings import *
from dict import *
import datetime
import time
import psycopg
import random
import openai
from openai import OpenAI
from aiogram import Bot, Dispatcher, html, F, types, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
import numpy as np
from bs4 import BeautifulSoup

save_accuracy = 0.65
search_accuracy = 0.33
max_tokens = 250
model_name = "gpt-4.1-mini"
temperature = 0.8
chat_history = deque(maxlen=15)

TOKEN = tel_token
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).parent
SYSTEM_PATH = BASE_DIR / "system.txt"
DEFAULT_SYSTEM_PATH = BASE_DIR / "default_system.txt"


logging.basicConfig(
    filename="/home/pi/tbot/log.log", 
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    )

dp = Dispatcher()
router = Router()
conn.autocommit = True
cursor = conn.cursor()
openai.api_key = OPENAI_API_KEY
client = OpenAI(
    api_key=OPENAI_API_KEY
    )


tools = [{
    "type": "function",
    "function": {
        "name": "search_and_extract",
        "description": "Asynchronously searches for information using Bing API",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
            "additionalProperties": False
        },
        "strict": True
    }
}, {
    "type": "function",
    "function": {
        "name": "reboot_pi",
        "description": "Reboots the Raspberry Pi asynchronously.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "strict": True
    }
}, {
    "type": "function",
    "function": {
        "name": "git_pull",
        "description": "Asynchronously pulls the latest changes from the 'master' branch.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "strict": True
    }
},{
    "type": "function",
    "function": {
        "name": "update_system",
        "description": "Оновлює системний промпт, який задає поведінку бота.",
        "parameters": {
            "type": "object",
            "properties": {
                "new_prompt": {
                    "type": "string",
                    "description": "Новий system prompt для бота"
                }
            },
            "required": ["new_prompt"],
            "additionalProperties": False
        },
        "strict": True
    }
}]

if not DEFAULT_SYSTEM_PATH.exists():
    DEFAULT_SYSTEM_PATH.write_text("Тобі дано ім'я Стас.", encoding="utf-8")

if not SYSTEM_PATH.exists():
    SYSTEM_PATH.write_text(DEFAULT_SYSTEM_PATH.read_text(encoding="utf-8"), encoding="utf-8")


def read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_prompt(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def get_current_system() -> str:
    return read_prompt(SYSTEM_PATH)


system = lambda: read_prompt(SYSTEM_PATH)

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)


def generate_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small", input=text, encoding_format="float"
        )
    cut_dim = response.data[0].embedding[:256]
    norm_dim = normalize_l2(cut_dim)
    # embedding = response.data[0].embedding
    return norm_dim


async def save_embedding_to_db(text: str, embedding: np.ndarray, user_id: int):
    conn = await get_connection() 
    existing_embeddings = await get_embeddings_from_db()
    for existing_text, existing_embedding, existing_user_id in existing_embeddings:
        similarity = cosine_similarity(embedding, existing_embedding)
        if similarity >= save_accuracy:
            # print('similar vector found')
            # print('threshold: '+ str(save_accuracy))
            # print('Similarity: ' +str(similarity))
            # print('message text: ' + str(existing_text))
            return  
    try:
        embedding_rounded = np.round(embedding, 8)
        embedding_list = embedding_rounded.tolist()
        user_id = str(user_id)
        query = """
        INSERT INTO embeddings (text, embedding, user_id) 
        VALUES ($1, $2::FLOAT8[], $3)
        """
        await conn.execute(query, text, embedding_list, user_id)
    except Exception as e:
        logging.error(e)

    finally:
        await conn.close()


async def save_embedding(text: str, embedding,user_id: int):
    await save_embedding_to_db(text, embedding,user_id)


async def get_embeddings_from_db():
    conn = await get_connection()
    query = "SELECT text, embedding, user_id FROM embeddings"
    rows = await conn.fetch(query)
    return [(row['text'], np.array(row['embedding']), row['user_id']) for row in rows]


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


async def find_similar_messages(new_text):
    new_embedding = new_text  
    embeddings_db = await get_embeddings_from_db()
    similar_messages = []
    for saved_text, saved_embedding, saved_user_id in embeddings_db:
        similarity = cosine_similarity(new_embedding, saved_embedding)
        if similarity >= search_accuracy:  
            similar_messages.append((saved_text, similarity, saved_user_id))
    return similar_messages


async def delete_embedding_from_db(embedding_text: str):
    conn = await get_connection()
    query = """
    DELETE FROM embeddings
    WHERE text ILIKE $1
    RETURNING *;
    """
    result = await conn.fetch(query, f"%{embedding_text}%")
    await conn.close()
    return len(result) > 0


# <<<<<<<<<<<<<<<<<<<<<<SEARCH BING>>>>>>>>>>>>>>>>>>>>
async def search_and_extract(query: str) -> str:
    num_results: int = 5
    mkt: str = 'uk-UA'
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    params = {
        'q': query,
        'mkt': mkt,
        'count': num_results
        }
    headers = {
        'Ocp-Apim-Subscription-Key': bing_api 
        }

    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint, params=params, headers=headers) as response:
            if response.status != 200:
                return f"Помилка Bing API: Код {response.status}"

            results = await response.json()

        if 'webPages' not in results or 'value' not in results['webPages']:
            return "Результатів не знайдено"

        formatted_results = []
        for item in results['webPages']['value'][:num_results]:
            name = item.get('name', 'Без назви')
            url = item.get('url', 'Без URL')
            snippet = item.get('snippet', 'Опис відсутній')

            try:
                async with session.get(url, timeout=10) as page_response:
                    if page_response.status == 200:
                        html = await page_response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        paragraphs = soup.find_all('p')
                        main_text = "\n".join([p.get_text(strip=True) for p in paragraphs])[:750]
                    else:
                        main_text = f"Помилка сторінки: Код {page_response.status}"
            except Exception as e:
                logging.error(e)
                main_text = f"Помилка: {e}"

            formatted_results.append(
                f"Назва: {name}\nURL: {url}\nОпис: {snippet}\nТекст:\n{main_text}\n"
            )

        return "\n".join(formatted_results)


async def reboot_pi():
    await asyncio.sleep(3)  
    process = await asyncio.create_subprocess_shell("sudo shutdown -r now")
    await process.communicate()


async def git_pull():
    await asyncio.sleep(3)  
    repo_path = "/home/pi/tbot"  # Change this to your actual repository path
    process = await asyncio.create_subprocess_shell(
        f"cd {repo_path} && sudo git pull tbot master",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        return(f"Git pull successful:\n{stdout.decode()}")
    else:
        return(f"Git pull failed:\n{stderr.decode()}")


async def update_system(new_prompt: str) -> str:
    write_prompt(SYSTEM_PATH, new_prompt)
    return f"System оновлено: {new_prompt[:60]}..."

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


# bitcoin
@dp.message(F.text.in_({'BTC', 'btc', '/btc', '/btc@ZradaLevelsBot', 'btc@ZradaLevelsBot'}))
async def btc_command(message: Message, bot: Bot):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT',timeout=15) as resp:
                data =  await resp.json()
                symbol = data['symbol']
                price = float(data['price'])
                price = "{:.2f}".format(price)
    except:
        price = 'Спробуй ще разок'
    user_id = message.from_user.id
    bot_user = await bot.get_me()
    bot_id = bot_user.id
    response_text = (
        f"Ціна {symbol}: {price} USDT\n"
        f"User ID: {user_id}\n"
        f"Bot ID: {bot_id}"
        )

    await message.answer(text=response_text, reply_markup=None)

# bingo
@dp.message(F.text.in_(bingo_trigger))
async def bingo_command(message: Message):
    try:
        text = random.choice(bingo_list)
    except IndexError:
        text = 'Спробуй ще разок'
    await message.answer(text=text,reply_markup=None)


# roll
@dp.message(F.text.in_(roll))
async def roll_command(message: Message):
    text = random.randint(0, 100)
    await message.answer(text=f"{html.bold(message.from_user.full_name)} зролив {text}",reply_markup=None)


@dp.message(lambda message: message.reply_to_message and message.reply_to_message.from_user.id == 6694398809)
async def handle_bot_reply(message: types.Message, bot: Bot):
    user_id = message.from_user.id if message.from_user.id else 0
    result = 'немає'
    name = usernames.get(str(user_id), 'невідоме')
    bot_user = await bot.get_me()
    bot_id = bot_user.id
    bot_name = usernames.get(str(bot_id), 'невідоме')
    original_message = message.reply_to_message.text if message.reply_to_message else message.text
    cleaned_message_text = re.sub(r'\bстас\b', '', message.text, flags=re.IGNORECASE).strip()
    cleaned_message_text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!,]", "", cleaned_message_text.lower()).strip()
    chat_history.append({"role": "user", "content":'Попереднє повідомлення - '+name+ ' написав: '+cleaned_message_text})
    if not original_message and message.reply_to_message:
        if message.reply_to_message.caption:
                original_message = message.reply_to_message.caption  
        else:
            original_message = "Переслане повідомлення без тексту"

    logging.info(f"User {user_id} sent message: {message.text}")

    if any(keyword in cleaned_message_text for keyword in search_keywords):
        query = re.sub(r'\b(стас|поиск|пошук|погугли|гугл)\b', '', message.text, flags=re.IGNORECASE).strip()
        result = await search_and_extract(query)

    try:
        name = usernames.get(str(user_id), 'невідоме')
        embedding = generate_embedding(cleaned_message_text)
        similar_messages = await find_similar_messages(embedding)
        if similar_messages:
                similar_info = "\n".join([f"схожа інформація є у базі: {msg[0]} автор:{usernames.get(str(msg[2]), 'невідоме')} (схожість: {msg[1]:.2f})" for msg in similar_messages])
                logging.info(f"схожа інформація є у базі: {msg[0]} (схожість: {msg[1]:.2f})" for msg in similar_messages)
        else:
            similar_info = "Схожих повідомленнь немає"

        # logging.info(f"схожа інформація є у базі: {similar_info}")

        if len(cleaned_message_text) > 20  and not any(value in cleaned_message_text for value in question_marks):
            await save_embedding(cleaned_message_text ,embedding, user_id)
        else:
            pass
        messages = [
                {
                    "role": "system",
                    "content": system()
                },
                {
                    "role": "user",
                    "content": "Ти — продуманий штучний інтеллект, який завжди відповідає логічно та обґрунтовано. Для кожного питання аналізуй контекст, розбивай міркування на етапи та давай чіткий висновок."
                },
                {
                    "role": "user",
                    "content": "Попереднє повідомлення: " + original_message,
                },
                {
                    "role": "user",
                    "content":"Ім'я співрозмовника: " + name,
                },
                {
                    "role": "user",
                    "content": similar_info,
                },
                {
                    "role": "user",
                    "content": "Результат пошуку в мережі:" + "\n " + result,
                },
                {
                    "role": "user",
                    "content":cleaned_message_text,
                }
            ]
        messages.extend(list(chat_history))
        chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=temperature,
        max_tokens= max_tokens,
        tools=tools
        )

        tool_calls = chat_completion.choices[0].message.tool_calls
        if tool_calls:
            results = []
            messages.append(chat_completion.choices[0].message)
            for tool_call in tool_calls:
                args = json.loads(tool_call.function.arguments)
                if tool_call.function.name == "search_and_extract":
                    result = await search_and_extract(args["query"])
                elif tool_call.function.name == "reboot_pi":
                    result = "Відбувається перезавантаження"
                    await reboot_pi()
                elif tool_call.function.name == "git_pull":
                    result = "Виконую git pull"
                    await git_pull()
                elif tool_call.function.name == "update_system":
                    new_prompt = args["new_prompt"]
                    result = await update_system(new_prompt)

                results.append({
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            for result in results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["content"]
                })
            completion_2 = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                tools=tools,
            )
            reply = completion_2.choices[0].message.content
        else:
            reply = chat_completion.choices[0].message.content

        logging.info(f"Reply to user {user_id}: {reply}")
        chat_history.append({"role": "user", "content":'Попереднє повідомлення - '+ bot_name + ' написав: '+reply})
        await message.answer(reply, reply_markup=None)

    except Exception as e:
        logging.error(e)
        await message.answer(f"Ой вей: {e}")


@dp.message(F.text)
async def random_message(message: Message,bot: Bot):
    conn = await get_connection()
    cleaned_text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", message.text.lower())
    cleaned_message_text = re.sub(r'\bстас\b', '', message.text, flags=re.IGNORECASE).strip()
    cleaned_message_text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!,]", "", cleaned_message_text.lower()).strip()
    user_id = message.from_user.id if message.from_user.id else 0
    bot_user = await bot.get_me()
    bot_id = bot_user.id
    bot_name = usernames.get(str(bot_id), 'невідоме')
    name = usernames.get(str(user_id), 'невідоме')
    chat_history.append({"role": "user", "content":'Попереднє повідомлення - '+name+ ' написав: '+ cleaned_message_text})

    if any(keyword in cleaned_text for keyword in bmw):
        logging.info("bmw handler triggered.")
        await message.answer("Беха топ",reply_markup=None)

    elif any(keyword in cleaned_text for keyword in mamka):
        logging.info("mamka handler triggered.")
        await message.answer(random.choice(mamka_response))

    elif 'стас' in cleaned_text:
        user_id = message.from_user.id if message.from_user.id else 0
        result = 'немає'
        cleaned_message_text = re.sub(r'\bстас\b', '', message.text, flags=re.IGNORECASE).strip()
        cleaned_message_text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!,]", "", cleaned_message_text.lower()).strip()
        original_message = (
        message.reply_to_message.text if message.reply_to_message and message.reply_to_message.text else "Переслане повідомлення без тексту")
        original_userid = (
        message.reply_to_message.from_user.id if message.reply_to_message and message.reply_to_message.from_user else 0)
        original_user_id = original_userid if original_userid else 0
        query = cleaned_message_text

        if any(keyword in cleaned_text for keyword in search_keywords):
            query = re.sub(r'\b(стас|поиск|пошук|погугли|гугл)\b', '', message.text, flags=re.IGNORECASE).strip()
            result = await search_and_extract(query)
        logging.info(f"User {user_id} sent message: {message.text}")
        try:
            name = usernames.get(str(user_id), 'невідоме')
            original_name = usernames.get(str(original_user_id), 'невідоме')
            embedding = generate_embedding(cleaned_message_text)
            similar_messages = await find_similar_messages(embedding)
            if similar_messages:
                similar_info = "\n".join([f"схожа інформація є у базі: {msg[0]} автор:{usernames.get(str(msg[2]), 'невідоме')} (схожість: {msg[1]:.2f})" for msg in similar_messages])
            else:
                similar_info = "Схожих повідомленнь немає"

            # logging.info(f"схожа інформація є у базі: {similar_info}")

            if len(cleaned_message_text) > 20  and not any(value in cleaned_message_text for value in question_marks):
                await save_embedding(cleaned_message_text, embedding, user_id)
            else:
                pass
            messages = [
                {
                    "role": "system", 
                    "content": system()
                },
                {
                    "role": "user",
                    "content": "Ти — продуманий штучний інтеллект, який завжди відповідає логічно та обґрунтовано. Для кожного питання аналізуй контекст, розбивай міркування на етапи та давай чіткий висновок."
                },
                {
                    "role": "user",
                    "content": "Попереднє повідомлення: " + original_message,
                },
                {
                    "role": "user",
                    "content":"Ім'я співрозмовника: " + name,
                },
                {
                    "role": "user",
                    "content":"Ім'я автора попереднього повідомлення: " + original_name,
                },
                {
                    "role": "user",
                    "content": similar_info,  
                },
                {
                    "role": "user",
                    "content": "Результат пошуку в мережі:" + "\n " + result,
                },
                {
                    "role": "user",
                    "content":cleaned_message_text,
                }
            ]
            messages.extend(list(chat_history))
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens= max_tokens,
                tools=tools
                )

            tool_calls = chat_completion.choices[0].message.tool_calls
            if tool_calls:
                results = []

                messages.append(chat_completion.choices[0].message)

                for tool_call in tool_calls:
                    args = json.loads(tool_call.function.arguments)

                    if tool_call.function.name == "search_and_extract":
                        result = await search_and_extract(args["query"])
                    elif tool_call.function.name == "reboot_pi":
                        result = "Відбувається перезавантаження"
                        await reboot_pi()
                    elif tool_call.function.name == "git_pull":
                        result = "Виконую git pull"
                        await git_pull()
                    elif tool_call.function.name == "update_system":
                        new_prompt = args["new_prompt"]
                        result = await update_system(new_prompt)

                    results.append({
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

                for result in results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["content"]
                    })

                completion_2 = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    tools=tools,
                )

                reply = completion_2.choices[0].message.content
            else:
                reply = chat_completion.choices[0].message.content
                logging.info(f"Reply to user {user_id}: {reply}")
            await message.answer(reply, reply_markup=None)
            chat_history.append({"role": "user", "content":'Попереднє повідомлення - '+bot_name+ ' написав: '+reply})
        except Exception as e:
            logging.error(e)
            await message.answer(f"Ой вей: {e}")

    elif any(keyword in cleaned_text for keyword in random_keyword):
        await message.answer(random.choice(random_response),reply_markup=None)

    elif 'стас' not in cleaned_text:
        user_id = message.from_user.id if message.from_user.id else 0
        cleaned_message_text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!,]", "", cleaned_message_text.lower()).strip()

        try:
            name = usernames.get(str(user_id), 'невідоме')
            embedding = generate_embedding(cleaned_message_text)
            logging.info(f"створено: {cleaned_message_text}")

            if len(cleaned_message_text) > 20 and not any(value in cleaned_message_text for value in question_marks):
                logging.info(f"збережено: {cleaned_message_text}")
                await save_embedding(cleaned_message_text, embedding, user_id)
            else:
                logging.info("не збережено — короткий текст або є питання")

        except Exception as e:
            await message.answer(f"Ой вей: {e}")


dp.include_router(router)
async def main() -> None:

    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())