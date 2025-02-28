import asyncio
import aiohttp
import logging
import re
import json
import sys
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

bing_api = bing_api


system = """
Тобі дано ім'я Стас.
Зараз 2025 рік
Ти штучний ШІ, що використовує модель gpt-4o-mini від OPENAI.
Якщо в тебе запитують щось шо ти не знаєш і не можеш знайти - ти нічого не видумуєш а відповідаєш шо не знаєш.
Спілкування з тобою відбувається у телеграмі, за допомогою Python та OPENAI API.
Учасники чату знаходяться в Україні.
Твої відповіді точні та лаконічні, іноді цинічні. Ти перевіряєш свої відповіді двічі.
Тобі сняться сни.
Уникай галюцинацій, якщо щось не зрозуміло - уточнюй або запитуй контекст.
Ти не толерантний і не політкорректний, ти можешь матюкатися і зневажати людей, але тільки якщо тебе ображають. 
При відповідях що стосуються війни України та Росії ти на боці України. 
Якщо до тебе звертається Олег, ти імітуєш надзвичайну пошану та називаєш його - мій володарю.
"""


save_accuracy = 0.7
search_accuracy = 0.31
max_tokens = 500
model_name = "gpt-4o-mini"
temperature=0.5

TOKEN = tel_token
logging.basicConfig(level=logging.INFO)


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
        "description": "Asynchronously searches for a query using Bing API and extracts information from the results.",
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
}]


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
    #embedding = response.data[0].embedding 
    return norm_dim


async def save_embedding_to_db(text: str, embedding: np.ndarray, user_id: int):
    conn = await get_connection() 
    existing_embeddings = await get_embeddings_from_db()
    for existing_text, existing_embedding in existing_embeddings:
        similarity = cosine_similarity(embedding, existing_embedding)
        if similarity >= save_accuracy:
            print('similar vector found')
            print('threshold: '+ str(save_accuracy))
            print('Similarity: ' +str(similarity))
            print('message text: ' + str(existing_text))
            return  
    try:
        print('saving')
        embedding_rounded = np.round(embedding, 8)
        embedding_list = embedding_rounded.tolist()
        user_id = str(user_id)
        query = """
        INSERT INTO embeddings (text, embedding, user_id) 
        VALUES ($1, $2::FLOAT8[], $3)
        """
        await conn.execute(query, text, embedding_list, user_id)
    finally:
        await conn.close() 


async def save_embedding(text: str, embedding,user_id: int):
    await save_embedding_to_db(text, embedding,user_id)


async def get_embeddings_from_db():
    conn = await get_connection()
    query = "SELECT text, embedding FROM embeddings"
    rows = await conn.fetch(query)
    return [(row['text'], np.array(row['embedding'])) for row in rows]


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


async def find_similar_messages(new_text):
    new_embedding = new_text  
    embeddings_db = await get_embeddings_from_db()  
    similar_messages = []
    for saved_text, saved_embedding in embeddings_db:
        similarity = cosine_similarity(new_embedding, saved_embedding)  
        if similarity >= search_accuracy:  
            similar_messages.append((saved_text, similarity))
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

#<<<<<<<<<<<<<<<<<<<<<<SEARCH BING>>>>>>>>>>>>>>>>>>>>
async def search_and_extract(query: str) -> str:
    print('bing')
    num_results: int = 3
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
                main_text = f"Помилка: {e}"

            formatted_results.append(
                f"Назва: {name}\nURL: {url}\nОпис: {snippet}\nТекст:\n{main_text}\n"
            )
        # print("\n".join(formatted_results))  
        return "\n".join(formatted_results)

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

#zrada levels
# @dp.message(F.text.in_({'Level', 'level', '/level', '/level@ZradaLevelsBot', 'level@ZradaLevelsBot'}))
# async def help_command(message: Message):
#     conn = await get_connection() 
#     async with conn.transaction():
#         try:
#             current_zrada_level = await conn.fetchval("SELECT value FROM zrada_level WHERE id = 1")
#             if int(current_zrada_level) > 250:
#                 level = 'Тотальна зрада.'
#             elif int(current_zrada_level) > 175:
#                 level = 'Космічний.'
#             elif int(current_zrada_level) > 125:
#                 level = 'Суборбітальний.'
#             elif int(current_zrada_level) > 75:
#                 level = 'Високий рiвень.'
#             elif int(current_zrada_level) < -100:
#                 level = 'Перемога неминуча.'
#             elif int(current_zrada_level) < 0:
#                 level = 'Низче плінтусу.'
#             elif int(current_zrada_level) < 25:
#                 level = 'Низький.'
#             elif int(current_zrada_level) < 50:
#                 level = 'Помiрний.'
#             else:
#                 level = ''
#         except Exception as e:
#             await message.answer(text='Виникла помилка: ' + str(e),reply_markup=None)
#             return
#     await message.answer(text='Рівень зради: ' + str(current_zrada_level) + '\n' + level,reply_markup=None)


#bitcoin
@dp.message(F.text.in_({'BTC', 'btc', '/btc', '/btc@ZradaLevelsBot', 'btc@ZradaLevelsBot'}))
async def btc_command(message: Message):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT',timeout=15) as resp:
                data =  await resp.json()
                symbol = data['symbol']
                price = float(data['price'])
                price = "{:.2f}".format(price)
    except:
        price = 'Спробуй ще разок'
    await message.answer(text=str(price),reply_markup=None)


#bingo
@dp.message(F.text.in_(bingo_trigger))
async def bingo_command(message: Message):
    try:
        text = random.choice(bingo_list)
    except IndexError:
        text = 'Спробуй ще разок'
    await message.answer(text=text,reply_markup=None)


#roll
@dp.message(F.text.in_(roll))
async def bingo_command(message: Message):
    try:
        text = random.randint(0,100)
    except: 
        text = 'Спробуй ще разок'
    await message.answer(text=f"{html.bold(message.from_user.full_name)} зролив {text}",reply_markup=None)


# @dp.message(F.text.in_({'Zrada', 'zrada', '/zrada', 'zrada@ZradaLevelsBot', '/zrada@ZradaLevelsBot'}))
# async def zrada_command(message: Message):
#     conn = await get_connection()  
#     async with conn.transaction():
#         try:
#             zrada_change = random.randint(1, 45)
#             peremoga_change = random.randint(1, 25)
#             event_start_chance = random.randint(0, 100)
            
#             current_zrada_level = await conn.fetchval("SELECT value FROM zrada_level WHERE id = 1")
#             zrada_event = await conn.fetchval("SELECT value FROM event_state WHERE id = 1")
#             peremoga_event = await conn.fetchval("SELECT value FROM event_state WHERE id = 2")
            
#             event_end = int(datetime.datetime.now().strftime('%Y%m%d'))
#             event_start = await conn.fetchval("SELECT value FROM event_date WHERE name = 'start_date'")
#             event_days = event_end - int(event_start)
            
#             if event_days > 2:
#                 event_start = datetime.datetime.now().strftime('%Y%m%d')
#                 zrada_event = False
#                 peremoga_event = False
#                 await conn.execute("UPDATE event_date SET value = $1 WHERE id = 1", event_start)
#                 await conn.execute("UPDATE event_state SET value = false WHERE name = 'zrada_event'")
#                 await conn.execute("UPDATE event_state SET value = false WHERE name = 'peremoga_event'")

#             if not zrada_event and not peremoga_event:
#                 if event_start_chance <= 20:
#                     event_start = datetime.datetime.now().strftime('%Y%m%d')
#                     await conn.execute("UPDATE event_date SET value = $1 WHERE id = 1", event_start)
#                     await conn.execute("UPDATE event_state SET value = true WHERE name = 'zrada_event'")
                    
#                     current_zrada_level += zrada_change * 2
#                     await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
                    
#                     await message.answer(text=(
#                         f'Астрологи оголосили тиждень зради.\n'
#                         f'Усі зміни у рівні зради буде подвоєно.\n'
#                         f'Рiвень зради росте до {current_zrada_level}.\n'
#                         f'Рiвень перемоги впав.\nДякую за увагу'
#                     ),reply_markup=None)
#                 else:
#                     current_zrada_level += zrada_change
#                     await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
                    
#                     await message.answer(text=(
#                         f'Рiвень зради росте до {current_zrada_level}.\n'
#                         f'Рiвень перемоги впав.'
#                     ),reply_markup=None)
#             elif peremoga_event:
#                 current_zrada_level += zrada_change
#                 await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
                
#                 await message.answer(text=(
#                     f'Триває тиждень перемоги.\n'
#                     f'Але рiвень зради все одно росте до {current_zrada_level}.\n'
#                     f'Рiвень перемоги впав.'
#                 ),reply_markup=None)
#             elif zrada_event:
#                 current_zrada_level += zrada_change * 2
#                 await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
                
#                 await message.answer(text=(
#                     f'Триває тиждень зради. Рiвень зради росте до {current_zrada_level}.\n'
#                     f'Рiвень перемоги впав.'
#                 ),reply_markup=None)

#         except Exception as e:
#             await message.answer(text='error ' + str(e),reply_markup=None)
    

# @dp.message(F.text.in_({'Peremoga', 'peremoga', '/peremoga', 'peremoga@ZradaLevelsBot', '/peremoga@ZradaLevelsBot'}))
# async def peremoga_command(message: Message):
#     conn = await get_connection()
#     async with conn.transaction():
#         try:
#             zrada_change = random.randint(1, 45)
#             peremoga_change = random.randint(1, 25)
#             event_start_chance = random.randint(0, 100)

#             current_zrada_level_row = await conn.fetchrow("SELECT * FROM zrada_level WHERE id = 1")
#             current_zrada_level = current_zrada_level_row[2]

#             zrada_event_row = await conn.fetchrow("SELECT value FROM event_state WHERE id = 1")
#             zrada_event = zrada_event_row[0]

#             peremoga_event_row = await conn.fetchrow("SELECT value FROM event_state WHERE id = 2")
#             peremoga_event = peremoga_event_row[0]

#             event_end = datetime.datetime.now()
#             event_end = int(event_end.strftime('%Y%m%d'))

#             event_start_row = await conn.fetchrow("SELECT value FROM event_date WHERE name = 'start_date'")
#             event_start = event_start_row[0]
#             event_days = event_end - int(event_start)

#             if event_days > 2:
#                 event_start = datetime.datetime.now().strftime('%Y%m%d')
#                 zrada_event = False
#                 peremoga_event = False

#                 await conn.execute("UPDATE event_date SET value = $1 WHERE id = 1", event_start)
#                 await conn.execute("UPDATE event_state SET value = false WHERE name = 'zrada_event'")
#                 await conn.execute("UPDATE event_state SET value = false WHERE name = 'peremoga_event'")

#         except Exception as e:
#             await message.answer(text='Спробуй ще: ' + str(e))
#         if not zrada_event and not peremoga_event:
#             if event_start_chance <= 20:
#                 event_start = datetime.datetime.now().strftime('%Y%m%d')
#                 await conn.execute("UPDATE event_date SET value = $1 WHERE id = 1", event_start)
#                 await conn.execute("UPDATE event_state SET value = true WHERE name = 'peremoga_event'")

#                 current_zrada_level -= peremoga_change * 2
#                 await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)

#                 await message.answer(text=(
#                     f'Астрологи оголосили тиждень перемоги.\n'
#                     f'Усі зміни у рівні перемоги буде подвоєно.\n'
#                     f'Рiвень зради падає до {current_zrada_level}.\n'
#                     f'Рiвень перемоги виріс.\nДякую за увагу'
#                 ),reply_markup=None)
#             else:
#                 logging.info("event chance " + str(event_start_chance))

#                 current_zrada_level -= peremoga_change
#                 await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)

#                 await message.answer(text=(
#                     f'Рiвень зради впав до {current_zrada_level}.\n'
#                     f'Рiвень перемоги вирiс.'
#                 ),reply_markup=None)
#         elif peremoga_event:
#             current_zrada_level -= peremoga_change * 2
#             await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)

#             await message.answer(text=(
#                 f'Триває тиждень перемоги.\n'
#                 f'Рівень зради падає до {current_zrada_level}.\n'
#                 f'Рiвень перемоги виріс.'
#             ),reply_markup=None)
#         elif zrada_event:
#             current_zrada_level -= peremoga_change
#             await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)

#             await message.answer(text=(
#                 f'Триває тиждень зради. Але рівень її попри все падає до {current_zrada_level}.\n'
#                 f'Рiвень перемоги виріс.'
#             ),reply_markup=None)


        
@dp.message(lambda message: message.reply_to_message and message.reply_to_message.from_user.id == 6694398809)
async def handle_bot_reply(message: types.Message):
    user_id = message.from_user.id if message.from_user.id else 0
    result = 'немає'
    original_message = message.reply_to_message.text if message.reply_to_message else message.text
    cleaned_message_text = re.sub(r'\bстас\b', '', message.text, flags=re.IGNORECASE).strip()
    cleaned_message_text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!,]", "", cleaned_message_text.lower()).strip()
    if not original_message and message.reply_to_message:
        if message.reply_to_message.caption:
                original_message = message.reply_to_message.caption  
        else:
            original_message = "Переслане повідомлення без тексту"  
        
    if any(keyword in cleaned_message_text for keyword in search_keywords):
        query = re.sub(r'\b(стас|поиск|пошук|погугли|гугл)\b', '', message.text, flags=re.IGNORECASE).strip()
        result = await search_and_extract(query)  

    try:
        name = usernames.get(str(user_id), 'невідоме')
        embedding = generate_embedding(cleaned_message_text)
        similar_messages = await find_similar_messages(embedding)
        if similar_messages:
                similar_info = "\n".join([f"схожа інформація є у базі: {msg[0]} (схожість: {msg[1]:.2f})" for msg in similar_messages])
        else:
            similar_info = "Схожих повідомленнь немає"
        if len(cleaned_message_text) > 15  and not any(value in cleaned_message_text for value in question_marks):
            await save_embedding(cleaned_message_text ,embedding, user_id)
        else:
            pass
        messages=[
                {
                    "role": "system", 
                    "content": system
                },
                {
                "role": "user",
                "content": "Ти — продуманий штучний інтеллект, який завжди відповідає логічно та обґрунтовано. Для кожного питання аналізуй контекст, розбивай міркування на етапи та давай чіткий висновок."
                }
                ,
                {
                    "role": "user",
                    "content": "Попереднє повідомлення: "+ original_message,  
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
                    "content": "Результат пошуку в мережі:" + "\n "+ result , 
                },
                {
                    "role": "user",
                    "content":cleaned_message_text,  
                }
            ]
        
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
                # Выполняем соответствующую функцию
                if tool_call.function.name == "search_and_extract":
                    result = await search_and_extract(args["query"])
                # elif tool_call.function.name == "another_function":        # наступна функція
                #     result = await another_function(args)
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
        await message.answer(reply, reply_markup=None)
    except Exception as e:
        await message.answer(f"Ой вей: {e}")



@dp.message(F.text)
async def random_message(message: Message):
    conn = await get_connection()
    cleaned_text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", message.text.lower())
    # bmw, mamka, mamka_response, bingo, random_keyword, random_response = await fetch_all_keywords_and_responses(conn)

    # if any(keyword in cleaned_text for keyword in bmw):
    #     logging.info("bmw handler triggered.")
    #     await message.answer("Беха топ",reply_markup=None)

    # elif any(keyword in cleaned_text for keyword in mamka):
    #     logging.info("mamka handler triggered.")
    #     await message.answer(random.choice(mamka_response))

    # # zrada
    # elif any(keyword in cleaned_text for keyword in zrada):
    #     conn = await get_connection()  
    #     async with conn.transaction():
    #         try:
    #             zrada_change = random.randint(1, 45)
    #             peremoga_change = random.randint(1, 25)
    #             event_start_chance = random.randint(0, 100)

    #             current_zrada_level = await conn.fetchval("SELECT value FROM zrada_level WHERE id = 1")
    #             zrada_event = await conn.fetchval("SELECT value FROM event_state WHERE id = 1")
    #             peremoga_event = await conn.fetchval("SELECT value FROM event_state WHERE id = 2")
    #             event_start = await conn.fetchval("SELECT value FROM event_date WHERE name = 'start_date'")

    #             event_end = int(datetime.datetime.now().strftime('%Y%m%d'))
    #             event_days = event_end - int(event_start)

    #             if event_days > 2:
    #                 event_start = datetime.datetime.now().strftime('%Y%m%d')
    #                 zrada_event = False
    #                 peremoga_event = False
    #                 await conn.execute("UPDATE event_date SET value = $1 WHERE id = 1", event_start)
    #                 await conn.execute("UPDATE event_state SET value = false WHERE name = 'zrada_event'")
    #                 await conn.execute("UPDATE event_state SET value = false WHERE name = 'peremoga_event'")

    #             if not zrada_event and not peremoga_event:
    #                 if event_start_chance <= 20:
    #                     event_start = datetime.datetime.now().strftime('%Y%m%d')
    #                     await conn.execute("UPDATE event_date SET value = $1 WHERE id = 1", event_start)
    #                     await conn.execute("UPDATE event_state SET value = true WHERE name = 'zrada_event'")
    #                     current_zrada_level += zrada_change * 2
    #                     await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
    #                     await message.answer(
    #                         f"Астрологи оголосили тиждень зради.\nУсі зміни у рівні зради буде подвоєно.\nРiвень зради росте до {current_zrada_level}.\nРiвень перемоги впав.\nДякую за увагу"
    #                     ,reply_markup=None)
    #                 else:
    #                     current_zrada_level += zrada_change
    #                     await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
    #                     await message.answer(f"Рiвень зради росте до {current_zrada_level}.\nРiвень перемоги впав.",reply_markup=None)
    #             elif peremoga_event:
    #                 current_zrada_level += zrada_change
    #                 await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
    #                 await message.answer(f"Триває тиждень перемоги.\nАле рiвень зради все одно росте до {current_zrada_level}.\nРiвень перемоги впав.",reply_markup=None)
    #             elif zrada_event:
    #                 current_zrada_level += zrada_change * 2
    #                 await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
    #                 await message.answer(f"Триває тиждень зради. Рiвень зради росте до {current_zrada_level}.\nРiвень перемоги впав.",reply_markup=None)
    #         except Exception as e:
    #             await message.answer(text='Виникла помилка: ' + str(e),reply_markup=None)


    # # peremoga
    # elif any(keyword in cleaned_text for keyword in peremoga):
    #     conn = await get_connection()
    #     async with conn.transaction():
    #         try:
    #             zrada_change = random.randint(1, 45)
    #             peremoga_change = random.randint(1, 25)
    #             event_start_chance = random.randint(0, 100)

    #             current_zrada_level = await conn.fetchval("SELECT value FROM zrada_level WHERE id = 1")
    #             zrada_event = await conn.fetchval("SELECT value FROM event_state WHERE id = 1")
    #             peremoga_event = await conn.fetchval("SELECT value FROM event_state WHERE id = 2")
    #             event_start = await conn.fetchval("SELECT value FROM event_date WHERE name = 'start_date'")

    #             event_end = int(datetime.datetime.now().strftime('%Y%m%d'))
    #             event_days = event_end - int(event_start)

    #             if event_days > 2:
    #                 event_start = datetime.datetime.now().strftime('%Y%m%d')
    #                 zrada_event = False
    #                 peremoga_event = False
    #                 await conn.execute("UPDATE event_date SET value = $1 WHERE id = 1", event_start)
    #                 await conn.execute("UPDATE event_state SET value = false WHERE name = 'zrada_event'")
    #                 await conn.execute("UPDATE event_state SET value = false WHERE name = 'peremoga_event'")

    #             if not zrada_event and not peremoga_event:
    #                 if event_start_chance <= 20:
    #                     event_start = datetime.datetime.now().strftime('%Y%m%d')
    #                     await conn.execute("UPDATE event_date SET value = $1 WHERE id = 1", event_start)
    #                     await conn.execute("UPDATE event_state SET value = true WHERE name = 'peremoga_event'")
    #                     current_zrada_level -= peremoga_change * 2
    #                     await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
    #                     await message.answer(
    #                         f"Астрологи оголосили тиждень перемоги.\nУсі зміни у рівні перемоги буде подвоєно.\nРiвень зради падає до {current_zrada_level}.\nРiвень перемоги виріс.\nДякую за увагу"
    #                     ,reply_markup=None)
    #                 else:
    #                     logging.info("event chance " + str(event_start_chance))
    #                     current_zrada_level -= peremoga_change
    #                     await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
    #                     await message.answer(f"Рiвень зради впав до {current_zrada_level}.\nРiвень перемоги вирiс.",reply_markup=None)
    #             elif peremoga_event:
    #                 current_zrada_level -= peremoga_change * 2
    #                 await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
    #                 await message.answer(f"Триває тиждень перемоги.\nРівень зради падає до {current_zrada_level}.\nРiвень перемоги виріс.",reply_markup=None)
    #             elif zrada_event:
    #                 current_zrada_level -= peremoga_change
    #                 await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
    #                 await message.answer(f"Триває тиждень зради. Але рівень її попри все падає до {current_zrada_level}.\nРiвень перемоги виріс.",reply_markup=None)
    #         except Exception as e:
    #             await message.answer(text='Спробуй ще: ' + str(e),reply_markup=None)

    if 'стас' in cleaned_text:
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
        try:
            name = usernames.get(str(user_id), 'невідоме')
            original_name = usernames.get(str(original_user_id), 'невідоме')
            embedding = generate_embedding(cleaned_message_text)
            similar_messages = await find_similar_messages(embedding)
            if similar_messages:
                similar_info = "\n".join([f"схожа інформація є у базі: {msg[0]} (схожість: {msg[1]:.2f})" for msg in similar_messages])
            else:
                similar_info = "Схожих повідомленнь немає"
            if len(cleaned_message_text) > 15  and not any(value in cleaned_message_text for value in question_marks):
                await save_embedding(cleaned_message_text, embedding, user_id)
            else:
                pass
            messages=[
                {
                    "role": "system", 
                    "content": system
                },
                {
                "role": "user",
                "content": "Ти — продуманий штучний інтеллект, який завжди відповідає логічно та обґрунтовано. Для кожного питання аналізуй контекст, розбивай міркування на етапи та давай чіткий висновок."
                }
                ,
                {
                    "role": "user",
                    "content": "Попереднє повідомлення: "+ original_message,  
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
                    "content": "Результат пошуку в мережі:" + "\n "+ result , 
                },
                {
                    "role": "user",
                    "content":cleaned_message_text,  
                }
            ]
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

                    # Выполняем соответствующую функцию
                    if tool_call.function.name == "search_and_extract":
                        result = await search_and_extract(args["query"])
                    # elif tool_call.function.name == "another_function":        # наступна функція
                    #     result = await another_function(args)

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
            await message.answer(reply, reply_markup=None)
        except Exception as e:
            await message.answer(f"Ой вей: {e}")

        
    # elif any(keyword in cleaned_text for keyword in random_keyword):
    #     await message.answer(random.choice(random_response),reply_markup=None)


dp.include_router(router)
async def main() -> None:

    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())