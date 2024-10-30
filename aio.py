import asyncio
import aiohttp
import logging
import re
import sys
from os import getenv
from settings import *
from dict import *
import datetime
import time
import psycopg
import random

from aiogram import Bot, Dispatcher, html, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.filters import Command
from aiogram.types import Message

from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

# Создание кнопок
button_btc = KeyboardButton(text="📈 BTC")
button_zrada = KeyboardButton(text="⚔️ Zrada")
button_peremoga = KeyboardButton(text="🏆 Peremoga")
button_bingo = KeyboardButton(text="🎲 Bingo")
button_level = KeyboardButton(text="📊 Level")
button_roll = KeyboardButton(text="🎲 Roll") 

keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [button_btc, button_zrada],
        [button_peremoga, button_bingo,button_level,button_roll]
    ],
    resize_keyboard=True
)

zrada = ['зрада','zrada']
peremoga = ['перемога','peremoga','перемога!']

# Bot token can be obtained via https://t.me/BotFather
TOKEN = tel_token
logging.basicConfig(level=logging.INFO)
# All handlers should be attached to the Router (or Dispatcher)

dp = Dispatcher()
cursor = conn.cursor()
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

def fetch_keywords_and_responses():
    cursor = conn.cursor()
    cursor.execute("SELECT keyword FROM keywords WHERE category = 'bmw'")
    bmw = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT keyword FROM keywords WHERE category = 'mamka'")
    mamka = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT response FROM responses WHERE category = 'mamka' ")
    mamka_response = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT response FROM responses WHERE category = 'bingo' ")
    bingo = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT keyword FROM keywords WHERE category = 'politics' ")
    random_keyword = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT response FROM responses WHERE category = 'politics' ")
    random_response = [row[0] for row in cursor.fetchall()]
    conn.commit()
    
   

    return bmw, mamka, mamka_response, bingo, random_keyword, random_response





#zrada levels
@dp.message(F.text.in_({'📊 Level', 'level', '/level', '/level@ZradaLevelsBot', 'level@ZradaLevelsBot'}))
async def help_command(message: Message):
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM zrada_level WHERE id = 1")
        current_zrada_level = cursor.fetchone()[2]
        if int(current_zrada_level) > 250:
            level = 'Тотальна зрада.'
        elif current_zrada_level > 175:
            level = 'Космічний.'
        elif current_zrada_level > 125:
            level = 'Суборбітальний.'
        elif current_zrada_level > 75:
            level = 'Високий рiвень.'
    
        elif int(current_zrada_level) <-100:
            level = 'Перемога неминуча.'
        elif int(current_zrada_level) <0:
            level = 'Низче плінтусу.'
        elif int(current_zrada_level) <25:
            level = 'Низький.'
        elif int(current_zrada_level) <50:
            level = 'Помiрний.'
        else:
            level = ''

    except:
        time.sleep(1.5)
        cursor.execute("SELECT * FROM zrada_level WHERE id = 1")
        current_zrada_level = cursor.fetchone()[2]

        if int(current_zrada_level) > 250:
            level = 'Тотальна зрада.'
        elif current_zrada_level > 175:
            level = 'Космічний.'
        elif current_zrada_level > 125:
            level = 'Суборбітальний.'
        elif current_zrada_level > 75:
            level = 'Високий рiвень.'
    
        elif int(current_zrada_level) <-100:
            level = 'Перемога неминуча.'
        elif int(current_zrada_level) <0:
            level = 'Низче плінтусу.'
        elif int(current_zrada_level) <25:
            level = 'Низький.'
        elif int(current_zrada_level) <50:
            level = 'Помiрний.'
        else:
            level = ''
    conn.commit()
    await message.answer(text='Рівень зради: ' + str(current_zrada_level)+'\n'+level, reply_markup=keyboard)

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
    await message.answer(text=str(price), reply_markup=keyboard)


#bingo
@dp.message(F.text.in_({'🎲 Bingo', 'bingo', '/bingo', '/bingo@ZradaLevelsBot', 'bingo@ZradaLevelsBot'}))
async def bingo_command(message: Message):
    bmw, mamka, mamka_response, bingo, random_keyword, random_response = fetch_keywords_and_responses()

    try:
        text = random.choice(bingo)
    except:
        price = 'Спробуй ще разок'
    await message.answer(text=text, reply_markup=keyboard)


#roll
@dp.message(F.text.in_({'🎲 Roll', 'roll', '/roll', '/roll@ZradaLevelsBot', 'roll@ZradaLevelsBot'}))
async def bingo_command(message: Message):
    try:
        text = random.randint(0,100)
    except: 
        text = 'Спробуй ще разок'
    await message.answer(text=f"{html.bold(message.from_user.full_name)} зролив {text}", reply_markup=keyboard)

#@dp.message(F.text.in_({'', '', ''}))
@dp.message(F.text.in_({'⚔️ Zrada', 'zrada', '/zrada', 'zrada@ZradaLevelsBot', '/zrada@ZradaLevelsBot'}))
async def zrada_command(message: Message):
    try:
        zrada_change = random.randint(1,45)
        peremoga_change = random.randint(1,25)
        event_start_chance = random.randint(0,100)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM zrada_level WHERE id = 1")
            current_zrada_level = cursor.fetchone()[2]
            cursor.execute("SELECT value FROM event_state WHERE id = 1 ")
            zrada_event = cursor.fetchone()[0]
            cursor.execute("SELECT value FROM event_state WHERE id = 2 ")
            peremoga_event = cursor.fetchone()[0]
            event_end = datetime.datetime.now()
            event_end = int(event_end.strftime('%Y%m%d'))
            cursor.execute("SELECT value from event_date WHERE name = 'start_date'")
            event_start = cursor.fetchone()[0]
            event_days = event_end-int(event_start)
            conn.commit()
                
        except Exception as e:
            await message.answer(text = 'error '+ e)
            conn.commit()
        if event_days >2:
            cursor = conn.cursor()
            event_start = datetime.datetime.now()
            event_start = event_start.strftime('%Y%m%d')
            zrada_event = False
            peremoga_event = False
            cursor.execute("UPDATE event_date set value = "+event_start+" WHERE id = 1")
            cursor.execute("UPDATE event_state SET value = false WHERE name = 'zrada_event' ")
            cursor.execute("UPDATE event_state SET value = false WHERE name = 'peremoga_event' ")
            conn.commit()

            
    except Exception as e:
        await message.answer(text = 'error 2 '+e)
        conn.commit()
       
    if zrada_event == False and peremoga_event == False:
        if event_start_chance <=20:
            cursor = conn.cursor()
            event_start = datetime.datetime.now()
            event_start = event_start.strftime('%Y%m%d')
            cursor.execute("UPDATE event_date SET value = "+event_start+"  WHERE id = 1 ")
            cursor.execute("UPDATE event_state SET value = true where name = 'zrada_event' ")
            current_zrada_level = int(current_zrada_level)+(zrada_change*2)
            current_zrada_level = str(current_zrada_level)
            cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
            cursor.execute("SELECT * FROM zrada_level WHERE id = 1")
            current_zrada_level = cursor.fetchone()[2]
            conn.commit()
            await message.answer(text = 'Астрологи оголосили тиждень зради.\nУсі зміни у рівні зради буде подвоєно.\nРiвень зради росте до '+str(current_zrada_level)+'.\nРiвень перемоги впав.\nДякую за увагу')
        elif event_start_chance >20:
            cursor = conn.cursor()
            current_zrada_level = int(current_zrada_level)+zrada_change
            current_zrada_level = str(current_zrada_level)
            cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
            cursor.execute("SELECT * FROM zrada_level WHERE id = 1")
            current_zrada_level = cursor.fetchone()[2]
            conn.commit()
            await message.answer(text = 'Рiвень зради росте до '+str(current_zrada_level)+'.\nРiвень перемоги впав.')

    elif peremoga_event == True:
        cursor = conn.cursor()
        current_zrada_level = int(current_zrada_level)+zrada_change
        current_zrada_level = str(current_zrada_level)
        cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
        conn.commit()
        await message.answer(text = 'Триває тиждень перемоги.\nАле рiвень зради все одно росте до '+current_zrada_level+'.\nРiвень перемоги впав.')
    elif zrada_event == True:
        cursor = conn.cursor()
        current_zrada_level = int(current_zrada_level)+(zrada_change*2)
        current_zrada_level = str(current_zrada_level)
        cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
        conn.commit()
        await message.answer(text = 'Триває тиждень зради.Рiвень зради росте до '+current_zrada_level+'.\nРiвень перемоги впав.')

@dp.message(F.text.in_({'🏆 Peremoga', 'peremoga', '/peremoga', 'peremoga@ZradaLevelsBot', '/peremoga@ZradaLevelsBot'}))
async def peremoga_command(message: Message):
    try:
        zrada_change = random.randint(1,45)
        peremoga_change = random.randint(1,25)
        event_start_chance = random.randint(0,100)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM zrada_level WHERE id = 1")
            current_zrada_level = cursor.fetchone()[2]
            cursor.execute("SELECT value FROM event_state WHERE id = 1 ")
            zrada_event = cursor.fetchone()[0]
            cursor.execute("SELECT value FROM event_state WHERE id = 2 ")
            peremoga_event = cursor.fetchone()[0]
            event_end = datetime.datetime.now()
            event_end = int(event_end.strftime('%Y%m%d'))
            cursor.execute("SELECT value from event_date WHERE name = 'start_date'")
            event_start = cursor.fetchone()[0]
            event_days = event_end-int(event_start)
            conn.commit()
        except Exception as e:
            await message.answer(text = 'error 3' +e)
            conn.commit()
        if event_days >2:
            cursor = conn.cursor()
            event_start = datetime.datetime.now()
            event_start = event_start.strftime('%Y%m%d')
            zrada_event = False
            peremoga_event = False
            cursor.execute("UPDATE event_date set value = "+event_start+" WHERE id = 1")
            cursor.execute("UPDATE event_state SET value = false WHERE name = 'zrada_event' ")
            cursor.execute("UPDATE event_state SET value = false WHERE name = 'peremoga_event' ")
            conn.commit()

    except Exception as e:
        await message.answer(text = 'Спробуй ще' + e)
        conn.commit()

    if zrada_event == False and peremoga_event == False:
        if event_start_chance <=20:
            cursor = conn.cursor()
            event_start = datetime.datetime.now()
            event_start = event_start.strftime('%Y%m%d')
            cursor.execute("UPDATE event_date SET value = "+event_start+"  WHERE id = 1 ")
            cursor.execute("UPDATE event_state SET value = true where name = 'peremoga_event' ")
            current_zrada_level = int(current_zrada_level)-(peremoga_change*2)
            current_zrada_level = str(current_zrada_level)
            cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
            conn.commit()
            await message.answer(text = 'Астрологи оголосили тиждень перемоги.\nУсі зміни у рівні перемоги буде подвоєно.\nРiвень зради падає до '+current_zrada_level+'.\nРiвень перемоги виріс.\nДякую за увагу')
        elif event_start_chance >20:
            cursor = conn.cursor()
            logging.info("event chance " +str(event_start_chance))
            current_zrada_level = int(current_zrada_level)-peremoga_change
            current_zrada_level = str(current_zrada_level)
            cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
            conn.commit()
            await message.answer(text = 'Рiвень зради впав до '+current_zrada_level+'.\nРiвень перемоги вирiс.')
        
                                    
    elif peremoga_event == True:
        cursor = conn.cursor()
        current_zrada_level = int(current_zrada_level)-(peremoga_change*2)
        current_zrada_level = str(current_zrada_level)
        cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
        conn.commit()
        await message.answer(text = 'Триває тиждень перемоги.\nРівень зради падає до '+current_zrada_level+'.\nРiвень перемоги виріс.')
    elif zrada_event == True:
        cursor = conn.cursor()
        current_zrada_level = int(current_zrada_level)-peremoga_change
        current_zrada_level = str(current_zrada_level)
        cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
        conn.commit()
        await message.answer(text = 'Триває тиждень зради.Але рівень її рівень попри все падає до '+current_zrada_level+'.\nРiвень перемоги виріс.')






@dp.message(F.text)
async def random_message(message: Message):
    cleaned_text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", message.text.lower())
    bmw, mamka, mamka_response, bingo, random_keyword, random_response = fetch_keywords_and_responses()

    if any(keyword in cleaned_text for keyword in bmw):
        logging.info("bmw handler triggered.")
        await message.answer(f"Беха топ")

    elif any(keyword in cleaned_text for keyword in mamka):
        logging.info("mamka handler triggered.")
        await message.answer(random.choice(mamka_response))

    #zrada
    elif any(keyword in cleaned_text for keyword in zrada):
        try:
            zrada_change = random.randint(1,45)
            peremoga_change = random.randint(1,25)
            event_start_chance = random.randint(0,100)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM zrada_level WHERE id = 1")
                current_zrada_level = cursor.fetchone()[2]
                cursor.execute("SELECT value FROM event_state WHERE id = 1 ")
                zrada_event = cursor.fetchone()[0]
                cursor.execute("SELECT value FROM event_state WHERE id = 2 ")
                peremoga_event = cursor.fetchone()[0]
                event_end = datetime.datetime.now()
                event_end = int(event_end.strftime('%Y%m%d'))
                cursor.execute("SELECT value from event_date WHERE name = 'start_date'")
                event_start = cursor.fetchone()[0]
                event_days = event_end-int(event_start)
                conn.commit()

            except Exception as e:
                await message.answer(text = 'error in F.text '+ e)
                conn.commit()

            if event_days >2:
                cursor = conn.cursor()
                event_start = datetime.datetime.now()
                event_start = event_start.strftime('%Y%m%d')
                zrada_event = False
                peremoga_event = False
                cursor.execute("UPDATE event_date set value = "+event_start+" WHERE id = 1")
                cursor.execute("UPDATE event_state SET value = false WHERE name = 'zrada_event' ")
                cursor.execute("UPDATE event_state SET value = false WHERE name = 'peremoga_event' ")
                conn.commit()
                

        except:
            await message.answer(text = 'error 1 in F.text ' +e)
           
        if zrada_event == False and peremoga_event == False:
            if event_start_chance <=20:
                cursor = conn.cursor()
                event_start = datetime.datetime.now()
                event_start = event_start.strftime('%Y%m%d')
                cursor.execute("UPDATE event_date SET value = "+event_start+"  WHERE id = 1 ")
                cursor.execute("UPDATE event_state SET value = true where name = 'zrada_event' ")
                current_zrada_level = int(current_zrada_level)+(zrada_change*2)
                current_zrada_level = str(current_zrada_level)
                cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                cursor.execute("SELECT * FROM zrada_level WHERE id = 1")
                current_zrada_level = cursor.fetchone()[2]
                conn.commit()
                await message.answer(text = 'Астрологи оголосили тиждень зради.\nУсі зміни у рівні зради буде подвоєно.\nРiвень зради росте до '+str(current_zrada_level)+'.\nРiвень перемоги впав.\nДякую за увагу')

            elif event_start_chance >20:
                cursor = conn.cursor()
                current_zrada_level = int(current_zrada_level)+zrada_change
                current_zrada_level = str(current_zrada_level)
                cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                cursor.execute("SELECT * FROM zrada_level WHERE id = 1")
                current_zrada_level = cursor.fetchone()[2]
                conn.commit()
                await message.answer(text = 'Рiвень зради росте до '+str(current_zrada_level)+'.\nРiвень перемоги впав.')

        elif peremoga_event == True:
            cursor = conn.cursor()
            current_zrada_level = int(current_zrada_level)+zrada_change
            current_zrada_level = str(current_zrada_level)
            cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
            conn.commit()
            await message.answer(text = 'Триває тиждень перемоги.\nАле рiвень зради все одно росте до '+current_zrada_level+'.\nРiвень перемоги впав.')

        elif zrada_event == True:
            cursor = conn.cursor()
            current_zrada_level = int(current_zrada_level)+(zrada_change*2)
            current_zrada_level = str(current_zrada_level)
            cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
            conn.commit()
            await message.answer(text = 'Триває тиждень зради.Рiвень зради росте до '+current_zrada_level+'.\nРiвень перемоги впав.')

        
    #peremoga
    elif any(keyword in cleaned_text for keyword in peremoga):
        try:
            zrada_change = random.randint(1,45)
            peremoga_change = random.randint(1,25)
            event_start_chance = random.randint(0,100)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM zrada_level WHERE id = 1")
                current_zrada_level = cursor.fetchone()[2]
                cursor.execute("SELECT value FROM event_state WHERE id = 1 ")
                zrada_event = cursor.fetchone()[0]
                cursor.execute("SELECT value FROM event_state WHERE id = 2 ")
                peremoga_event = cursor.fetchone()[0]
                event_end = datetime.datetime.now()
                event_end = int(event_end.strftime('%Y%m%d'))
                cursor.execute("SELECT value from event_date WHERE name = 'start_date'")
                event_start = cursor.fetchone()[0]
                event_days = event_end-int(event_start)
                conn.commit()

            except Exception as e:
                await message.answer(text = 'error in peremoga '+e)
                conn.commit()

            if event_days >2:
                cursor = conn.cursor()
                event_start = datetime.datetime.now()
                event_start = event_start.strftime('%Y%m%d')
                zrada_event = False
                peremoga_event = False
                cursor.execute("UPDATE event_date set value = "+event_start+" WHERE id = 1")
                cursor.execute("UPDATE event_state SET value = false WHERE name = 'zrada_event' ")
                cursor.execute("UPDATE event_state SET value = false WHERE name = 'peremoga_event' ")
                conn.commit()
        except Exception as e:
            await message.answer(text = 'Спробуй ще '+e)

        if zrada_event == False and peremoga_event == False:
            if event_start_chance <=20:
                cursor = conn.cursor()
                event_start = datetime.datetime.now()
                event_start = event_start.strftime('%Y%m%d')
                cursor.execute("UPDATE event_date SET value = "+event_start+"  WHERE id = 1 ")
                cursor.execute("UPDATE event_state SET value = true where name = 'peremoga_event' ")
                current_zrada_level = int(current_zrada_level)-(peremoga_change*2)
                current_zrada_level = str(current_zrada_level)
                cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                conn.commit()
                await message.answer(text = 'Астрологи оголосили тиждень перемоги.\nУсі зміни у рівні перемоги буде подвоєно.\nРiвень зради падає до '+current_zrada_level+'.\nРiвень перемоги виріс.\nДякую за увагу')

            elif event_start_chance >20:
                cursor = conn.cursor()
                logging.info("event chance " +str(event_start_chance))
                current_zrada_level = int(current_zrada_level)-peremoga_change
                current_zrada_level = str(current_zrada_level)
                cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                conn.commit()
                await message.answer(text = 'Рiвень зради впав до '+current_zrada_level+'.\nРiвень перемоги вирiс.')
            
                                        
        elif peremoga_event == True:
            cursor = conn.cursor()
            current_zrada_level = int(current_zrada_level)-(peremoga_change*2)
            current_zrada_level = str(current_zrada_level)
            cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
            conn.commit()
            await message.answer(text = 'Триває тиждень перемоги.\nРівень зради падає до '+current_zrada_level+'.\nРiвень перемоги виріс.')
        elif zrada_event == True:
            cursor = conn.cursor()
            current_zrada_level = int(current_zrada_level)-peremoga_change
            current_zrada_level = str(current_zrada_level)
            cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
            conn.commit()
            await message.answer(text = 'Триває тиждень зради.Але рівень її рівень попри все падає до '+current_zrada_level+'.\nРiвень перемоги виріс.')


        
    
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