import asyncio
import aiohttp
import logging
import re
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
from aiogram import Bot, Dispatcher, html, F, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

# button_btc = KeyboardButton(text="📈 BTC")
# button_zrada = KeyboardButton(text="⚔️ Zrada")
# button_peremoga = KeyboardButton(text="🏆 Peremoga")
# button_bingo = KeyboardButton(text="🎲 Bingo")
# button_level = KeyboardButton(text="📊 Level")
# button_roll = KeyboardButton(text="🎲 Roll") 

# keyboard = ReplyKeyboardMarkup(
#     keyboard=[
#         [
#         #button_btc, 
#          button_zrada],
#         [button_peremoga, button_bingo,button_level,button_roll]
#     ],
#     resize_keyboard=True
# )

zrada = ['зрада','zrada']
peremoga = ['перемога','peremoga','перемога!']

TOKEN = tel_token
logging.basicConfig(level=logging.INFO)
# All handlers should be attached to the Router (or Dispatcher)

dp = Dispatcher()
conn.autocommit = True
cursor = conn.cursor()

client = OpenAI(
    # This is the default and can be omitted
    api_key=OPENAI_API_KEY
)

# @dp.message(CommandStart())
# async def command_start_handler(message: Message) -> None:
#     """
#     This handler receives messages with `/start` command
#     """
#     # Most event objects have aliases for API methods that can be called in events' context
#     # For example if you want to answer to incoming message you can use `message.answer(...)` alias
#     # and the target chat will be passed to :ref:`aiogram.methods.send_message.SendMessage`
#     # method automatically or call API method directly via
#     # Bot instance: `bot.send_message(chat_id=message.chat.id, ...)`
#     await message.answer(f"Hello, {html.bold(message.from_user.full_name)}!")

async def fetch_all_keywords_and_responses(conn):
    try:
        rows = await conn.fetch("SELECT keyword, category FROM keywords UNION ALL SELECT response, category FROM responses")
        
        results = {
            'bmw': [],
            'mamka': [],
            'mamka_response': [],
            'bingo': [],
            'politics': [],
            'politics_response': []
        }

        for value, category in rows:
            if category in results:
                results[category].append(value)

        return (
            results['bmw'],
            results['mamka'],
            results['mamka_response'],
            results['bingo'],
            results['politics'],
            results['politics_response']
        )
    finally:
        await conn.close()



#zrada levels
@dp.message(F.text.in_({'📊 Level', 'level', '/level', '/level@ZradaLevelsBot', 'level@ZradaLevelsBot'}))
async def help_command(message: Message):
    conn = await get_connection() 
    async with conn.transaction():
        try:
            current_zrada_level = await conn.fetchval("SELECT value FROM zrada_level WHERE id = 1")
            if int(current_zrada_level) > 250:
                level = 'Тотальна зрада.'
            elif int(current_zrada_level) > 175:
                level = 'Космічний.'
            elif int(current_zrada_level) > 125:
                level = 'Суборбітальний.'
            elif int(current_zrada_level) > 75:
                level = 'Високий рiвень.'
            elif int(current_zrada_level) < -100:
                level = 'Перемога неминуча.'
            elif int(current_zrada_level) < 0:
                level = 'Низче плінтусу.'
            elif int(current_zrada_level) < 25:
                level = 'Низький.'
            elif int(current_zrada_level) < 50:
                level = 'Помiрний.'
            else:
                level = ''
        except Exception as e:
            await message.answer(text='Виникла помилка: ' + str(e))
            return
    await message.answer(text='Рівень зради: ' + str(current_zrada_level) + '\n' + level)


#bitcoin
@dp.message(F.text.in_({'📈 BTC', 'btc', '/btc', '/btc@ZradaLevelsBot', 'btc@ZradaLevelsBot'}))
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
    await message.answer(text=str(price))


#bingo
@dp.message(F.text.in_({'🎲 Bingo', 'bingo', '/bingo', '/bingo@ZradaLevelsBot', 'bingo@ZradaLevelsBot'}))
async def bingo_command(message: Message):
    conn = await get_connection()
    bmw, mamka, mamka_response, bingo, random_keyword, random_response = await fetch_all_keywords_and_responses(conn)
    try:
        text = random.choice(bingo)
    except IndexError:
        text = 'Спробуй ще разок'
    await message.answer(text=text)
 # , reply_markup=keyboard


#roll
@dp.message(F.text.in_({'🎲 Roll', 'roll', '/roll', '/roll@ZradaLevelsBot', 'roll@ZradaLevelsBot'}))
async def bingo_command(message: Message):
    try:
        text = random.randint(0,100)
    except: 
        text = 'Спробуй ще разок'
    await message.answer(text=f"{html.bold(message.from_user.full_name)} зролив {text}")

#@dp.message(F.text.in_({'', '', ''}))
@dp.message(F.text.in_({'⚔️ Zrada', 'zrada', '/zrada', 'zrada@ZradaLevelsBot', '/zrada@ZradaLevelsBot'}))
async def zrada_command(message: Message):
    conn = await get_connection()  
    async with conn.transaction():
        try:
            zrada_change = random.randint(1, 45)
            peremoga_change = random.randint(1, 25)
            event_start_chance = random.randint(0, 100)
            
            current_zrada_level = await conn.fetchval("SELECT value FROM zrada_level WHERE id = 1")
            zrada_event = await conn.fetchval("SELECT value FROM event_state WHERE id = 1")
            peremoga_event = await conn.fetchval("SELECT value FROM event_state WHERE id = 2")
            
            event_end = int(datetime.datetime.now().strftime('%Y%m%d'))
            event_start = await conn.fetchval("SELECT value FROM event_date WHERE name = 'start_date'")
            event_days = event_end - int(event_start)
            
            if event_days > 2:
                event_start = datetime.datetime.now().strftime('%Y%m%d')
                zrada_event = False
                peremoga_event = False
                await conn.execute("UPDATE event_date SET value = $1 WHERE id = 1", event_start)
                await conn.execute("UPDATE event_state SET value = false WHERE name = 'zrada_event'")
                await conn.execute("UPDATE event_state SET value = false WHERE name = 'peremoga_event'")

            if not zrada_event and not peremoga_event:
                if event_start_chance <= 20:
                    event_start = datetime.datetime.now().strftime('%Y%m%d')
                    await conn.execute("UPDATE event_date SET value = $1 WHERE id = 1", event_start)
                    await conn.execute("UPDATE event_state SET value = true WHERE name = 'zrada_event'")
                    
                    current_zrada_level += zrada_change * 2
                    await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
                    
                    await message.answer(text=(
                        f'Астрологи оголосили тиждень зради.\n'
                        f'Усі зміни у рівні зради буде подвоєно.\n'
                        f'Рiвень зради росте до {current_zrada_level}.\n'
                        f'Рiвень перемоги впав.\nДякую за увагу'
                    ))
                else:
                    current_zrada_level += zrada_change
                    await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
                    
                    await message.answer(text=(
                        f'Рiвень зради росте до {current_zrada_level}.\n'
                        f'Рiвень перемоги впав.'
                    ))
            elif peremoga_event:
                current_zrada_level += zrada_change
                await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
                
                await message.answer(text=(
                    f'Триває тиждень перемоги.\n'
                    f'Але рiвень зради все одно росте до {current_zrada_level}.\n'
                    f'Рiвень перемоги впав.'
                ))
            elif zrada_event:
                current_zrada_level += zrada_change * 2
                await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
                
                await message.answer(text=(
                    f'Триває тиждень зради. Рiвень зради росте до {current_zrada_level}.\n'
                    f'Рiвень перемоги впав.'
                ))

        except Exception as e:
            await message.answer(text='error ' + str(e))


    

@dp.message(F.text.in_({'🏆 Peremoga', 'peremoga', '/peremoga', 'peremoga@ZradaLevelsBot', '/peremoga@ZradaLevelsBot'}))
async def peremoga_command(message: Message):
    conn = await get_connection()
    async with conn.transaction():
        try:
            zrada_change = random.randint(1, 45)
            peremoga_change = random.randint(1, 25)
            event_start_chance = random.randint(0, 100)

            current_zrada_level_row = await conn.fetchrow("SELECT * FROM zrada_level WHERE id = 1")
            current_zrada_level = current_zrada_level_row[2]

            zrada_event_row = await conn.fetchrow("SELECT value FROM event_state WHERE id = 1")
            zrada_event = zrada_event_row[0]

            peremoga_event_row = await conn.fetchrow("SELECT value FROM event_state WHERE id = 2")
            peremoga_event = peremoga_event_row[0]

            event_end = datetime.datetime.now()
            event_end = int(event_end.strftime('%Y%m%d'))

            event_start_row = await conn.fetchrow("SELECT value FROM event_date WHERE name = 'start_date'")
            event_start = event_start_row[0]
            event_days = event_end - int(event_start)

            if event_days > 2:
                event_start = datetime.datetime.now().strftime('%Y%m%d')
                zrada_event = False
                peremoga_event = False

                await conn.execute("UPDATE event_date SET value = $1 WHERE id = 1", event_start)
                await conn.execute("UPDATE event_state SET value = false WHERE name = 'zrada_event'")
                await conn.execute("UPDATE event_state SET value = false WHERE name = 'peremoga_event'")

        except Exception as e:
            await message.answer(text='Спробуй ще: ' + str(e))
        if not zrada_event and not peremoga_event:
            if event_start_chance <= 20:
                event_start = datetime.datetime.now().strftime('%Y%m%d')
                await conn.execute("UPDATE event_date SET value = $1 WHERE id = 1", event_start)
                await conn.execute("UPDATE event_state SET value = true WHERE name = 'peremoga_event'")

                current_zrada_level -= peremoga_change * 2
                await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)

                await message.answer(text=(
                    f'Астрологи оголосили тиждень перемоги.\n'
                    f'Усі зміни у рівні перемоги буде подвоєно.\n'
                    f'Рiвень зради падає до {current_zrada_level}.\n'
                    f'Рiвень перемоги виріс.\nДякую за увагу'
                ))
            else:
                logging.info("event chance " + str(event_start_chance))

                current_zrada_level -= peremoga_change
                await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)

                await message.answer(text=(
                    f'Рiвень зради впав до {current_zrada_level}.\n'
                    f'Рiвень перемоги вирiс.'
                ))
        elif peremoga_event:
            current_zrada_level -= peremoga_change * 2
            await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)

            await message.answer(text=(
                f'Триває тиждень перемоги.\n'
                f'Рівень зради падає до {current_zrada_level}.\n'
                f'Рiвень перемоги виріс.'
            ))
        elif zrada_event:
            current_zrada_level -= peremoga_change
            await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)

            await message.answer(text=(
                f'Триває тиждень зради. Але рівень її попри все падає до {current_zrada_level}.\n'
                f'Рiвень перемоги виріс.'
            ))

@dp.message(F.text.in_({'ало','ало'}))
async def openai_command(message: Message):
    try:
        chat_completion = client.chat.completions.create(
        messages=[
        {
            "role": "user",
            "content": message.text,
        }
        ],
        model="gpt-3.5-turbo",
        )
        print(chat_completion)

        reply = chat_completion.choices[0].message.content
        
        await message.answer(reply)
    
    except Exception as e:
        print(chat_completion)
        if "429" in str(e):
            await message.answer("Слишком много запросов. Пожалуйста, попробуйте позже.")
        else:
            await message.answer(f"Произошла ошибка: {e}")
        
@dp.message(lambda message: message.reply_to_message and message.reply_to_message.from_user.id == 6694398809)
async def handle_bot_reply(message: types.Message):
    user_reply = message.text
    original_message = message.reply_to_message.text

    try:
        chat_completion = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[
                {
                    "role": "user",
                    "content": original_message,  # Оригинальное сообщение
                },
                {
                    "role": "user",
                    "content": user_reply,  # Ответ пользователя
                }
            ],
            model="gpt-4o-mini",
        )

        # Извлечение отв
        reply = chat_completion.choices[0].message.content
        await message.answer(reply)

    except Exception as e:
        await message.answer(f"Произошла ошибка: {e}")






@dp.message(F.text)
async def random_message(message: Message):
    conn = await get_connection()
    cleaned_text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", message.text.lower())
    bmw, mamka, mamka_response, bingo, random_keyword, random_response = await fetch_all_keywords_and_responses(conn)

    if any(keyword in cleaned_text for keyword in bmw):
        logging.info("bmw handler triggered.")
        await message.answer("Беха топ")

    elif any(keyword in cleaned_text for keyword in mamka):
        logging.info("mamka handler triggered.")
        await message.answer(random.choice(mamka_response))

    # zrada
    elif any(keyword in cleaned_text for keyword in zrada):
        conn = await get_connection()  
        async with conn.transaction():
            try:
                zrada_change = random.randint(1, 45)
                peremoga_change = random.randint(1, 25)
                event_start_chance = random.randint(0, 100)

                # Запрос текущего уровня зрады и состояния событий
                current_zrada_level, zrada_event, peremoga_event, event_start = await asyncio.gather(
                    conn.fetchval("SELECT value FROM zrada_level WHERE id = 1"),
                    conn.fetchval("SELECT value FROM event_state WHERE id = 1"),
                    conn.fetchval("SELECT value FROM event_state WHERE id = 2"),
                    conn.fetchval("SELECT value FROM event_date WHERE name = 'start_date'")
                )

                event_end = int(datetime.datetime.now().strftime('%Y%m%d'))
                event_days = event_end - int(event_start)

                if event_days > 2:
                    event_start = datetime.datetime.now().strftime('%Y%m%d')
                    zrada_event = False
                    peremoga_event = False
                    await conn.execute("UPDATE event_date SET value = $1 WHERE id = 1", event_start)
                    await conn.execute("UPDATE event_state SET value = false WHERE name = 'zrada_event'")
                    await conn.execute("UPDATE event_state SET value = false WHERE name = 'peremoga_event'")

                if not zrada_event and not peremoga_event:
                    if event_start_chance <= 20:
                        event_start = datetime.datetime.now().strftime('%Y%m%d')
                        await conn.execute("UPDATE event_date SET value = $1 WHERE id = 1", event_start)
                        await conn.execute("UPDATE event_state SET value = true WHERE name = 'zrada_event'")
                        current_zrada_level += zrada_change * 2
                        await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
                        await message.answer(
                            f"Астрологи оголосили тиждень зради.\nУсі зміни у рівні зради буде подвоєно.\nРiвень зради росте до {current_zrada_level}.\nРiвень перемоги впав.\nДякую за увагу"
                        )
                    else:
                        current_zrada_level += zrada_change
                        await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
                        await message.answer(f"Рiвень зради росте до {current_zrada_level}.\nРiвень перемоги впав.")
                elif peremoga_event:
                    current_zrada_level += zrada_change
                    await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
                    await message.answer(f"Триває тиждень перемоги.\nАле рiвень зради все одно росте до {current_zrada_level}.\nРiвень перемоги впав.")
                elif zrada_event:
                    current_zrada_level += zrada_change * 2
                    await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
                    await message.answer(f"Триває тиждень зради. Рiвень зради росте до {current_zrada_level}.\nРiвень перемоги впав.")
            except Exception as e:
                await message.answer(text='Виникла помилка: ' + str(e))


    # peremoga
    elif any(keyword in cleaned_text for keyword in peremoga):
        conn = await get_connection()
        async with conn.transaction():
            try:
                zrada_change = random.randint(1, 45)
                peremoga_change = random.randint(1, 25)
                event_start_chance = random.randint(0, 100)

                current_zrada_level, zrada_event, peremoga_event, event_start = await asyncio.gather(
                    conn.fetchval("SELECT value FROM zrada_level WHERE id = 1"),
                    conn.fetchval("SELECT value FROM event_state WHERE id = 1"),
                    conn.fetchval("SELECT value FROM event_state WHERE id = 2"),
                    conn.fetchval("SELECT value FROM event_date WHERE name = 'start_date'")
                )

                event_end = int(datetime.datetime.now().strftime('%Y%m%d'))
                event_days = event_end - int(event_start)

                if event_days > 2:
                    event_start = datetime.datetime.now().strftime('%Y%m%d')
                    zrada_event = False
                    peremoga_event = False
                    await conn.execute("UPDATE event_date SET value = $1 WHERE id = 1", event_start)
                    await conn.execute("UPDATE event_state SET value = false WHERE name = 'zrada_event'")
                    await conn.execute("UPDATE event_state SET value = false WHERE name = 'peremoga_event'")

                if not zrada_event and not peremoga_event:
                    if event_start_chance <= 20:
                        event_start = datetime.datetime.now().strftime('%Y%m%d')
                        await conn.execute("UPDATE event_date SET value = $1 WHERE id = 1", event_start)
                        await conn.execute("UPDATE event_state SET value = true WHERE name = 'peremoga_event'")
                        current_zrada_level -= peremoga_change * 2
                        await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
                        await message.answer(
                            f"Астрологи оголосили тиждень перемоги.\nУсі зміни у рівні перемоги буде подвоєно.\nРiвень зради падає до {current_zrada_level}.\nРiвень перемоги виріс.\nДякую за увагу"
                        )
                    else:
                        logging.info("event chance " + str(event_start_chance))
                        current_zrada_level -= peremoga_change
                        await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
                        await message.answer(f"Рiвень зради впав до {current_zrada_level}.\nРiвень перемоги вирiс.")
                elif peremoga_event:
                    current_zrada_level -= peremoga_change * 2
                    await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
                    await message.answer(f"Триває тиждень перемоги.\nРівень зради падає до {current_zrada_level}.\nРiвень перемоги виріс.")
                elif zrada_event:
                    current_zrada_level -= peremoga_change
                    await conn.execute("UPDATE zrada_level SET value = $1 WHERE id = 1", current_zrada_level)
                    await message.answer(f"Триває тиждень зради. Але рівень її попри все падає до {current_zrada_level}.\nРiвень перемоги виріс.")
            except Exception as e:
                await message.answer(text='Спробуй ще: ' + str(e))

    elif 'бот'  in cleaned_text:
        original_message = message.reply_to_message.text if message.reply_to_message else message.text
        if not original_message and message.reply_to_message:
            if message.reply_to_message.caption:
                original_message = message.reply_to_message.caption  # Используем заголовок медиа
        else:
            original_message = "Пересланное сообщение без текста."  # Сообщение для пользователя, если текст отсутствует
        try:
            chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": original_message,  # Передаем оригинальное сообщение
                },
                {
                    "role": "user",
                    "content": message.text,  # Передаем текст, который пользователь отправил
                }
            ],
            model="gpt-4o-mini",
            )
            reply = chat_completion.choices[0].message.content
            await message.answer(reply)
        except Exception as e:
            await message.answer(f"Произошла ошибка: {e}")

    elif any(keyword in cleaned_text for keyword in random_keyword):
        await message.answer(random.choice(random_response))
  




    

    
   


    #peremoga
    
        








    # Respond based on the keyword found in the message
    # if 'hello' in message.text.lower():
    #     await message.answer("Greetings!")
    # elif 'bye' in message.text.lower():
    #     await message.answer("Farewell!")
    # elif 'help' in message.text.lower():
    #     await message.answer("How can I assist you?")





# @dp.message()
# async def echo_handler(message: Message) -> None:
#     """
#     Handler will forward receive a message back to the sender

#     By default, message handler will handle all message types (like a text, photo, sticker etc.)
#     """
#     print(message)
#     try:
        
#         # Send a copy of the received message
#         await message.send_copy(chat_id=message.chat.id)
#     except TypeError:
#         # But not all the types is supported to be copied so need to handle it
#         await message.answer("Nice try!")


async def main() -> None:
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())