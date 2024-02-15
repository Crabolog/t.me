import requests as r
import json
import asyncio
import aiohttp
import random
import datetime
import time
import re
import psycopg2
from  dict import *
from settings import *
loop = asyncio.new_event_loop()


async def bot():
    update = None
    offset = 1
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM zrada_level WHERE id = 1")
    current_zrada_level = cursor.fetchone()[2]
   

    event_start = datetime.datetime.now() #datetime.datetime.strptime('2024-02-12 19:43:55.985354', '%Y-%m-%d %H:%M:%S.%f')
    event_end = datetime.datetime.now()
    event_start = int(event_start.strftime('%Y%m%d'))
    event_end = int(event_end.strftime('%Y%m%d'))
    event_days = event_end-event_start
    zrada_event = False
    peremoga_event = False

    while True:

        if event_days >1:
            event_start = datetime.datetime.now()
            event_start = int(event_start.strftime('%Y%m%d'))
            zrada_event = False
            peremoga_event = False
        elif event_days <=1:
            pass
        time.sleep(0.7)
        zrada_change = random.randint(1,45)
        peremoga_change = random.randint(1,25)
        event_start_chance = random.randint(0,100)
        try:
            
            async with aiohttp.ClientSession() as session:
                async with session.get(tel_api+tel_token+'/getUpdates?offset='+f"{offset}",timeout=5) as resp:
                    data =  await resp.json()
                    update_id = data['result'][-1]['update_id']
                    offset = data['result'][-1]['update_id']

                    if update_id == update:
                        time.sleep(1)
                        pass

                    elif update_id != update:
                        update = update_id
                        offset = update_id
                        user_id = data['result'][-1]['message']['from']['id']
                        chat_id = data['result'][-1]['message']['chat']['id']
                        username = data['result'][-1]['message']['from']['username']
                        message_id = data['result'][-1]['message']['message_id']
                        text = data['result'][-1]['message']['text']
                        text = text.lower()
                    
                        
                #status check    
                        if text in status:
                            current_zrada_level = int(current_zrada_level)
                            if int(current_zrada_level) > 250:
                                message = {'chat_id':chat_id, 'user_id':user_id,'text':'Рiвень зради: '+str(current_zrada_level)+'\nТотальна зрада.'}
                                await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                            
                            elif current_zrada_level > 175:
                                message = {'chat_id':chat_id, 'user_id':user_id,'text':'Рiвень зради: '+str(current_zrada_level)+'\nКосмічний.'}
                                await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                            elif current_zrada_level > 125:
                                message = {'chat_id':chat_id, 'user_id':user_id,'text':'Рiвень зради: '+str(current_zrada_level)+'\nСуборбітальний.'}
                                await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                            elif current_zrada_level > 50:
                                message = {'chat_id':chat_id, 'user_id':user_id,'text':'Рiвень зради: '+str(current_zrada_level)+'\nВисокий рiвень.'}
                                await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                            
                            elif int(current_zrada_level) <-100:
                                message = {'chat_id':chat_id, 'user_id':user_id,'text':'Рiвень зради: '+str(current_zrada_level)+'\nПеремога неминуча.'}
                                await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                            elif int(current_zrada_level) <0:
                                message = {'chat_id':chat_id, 'user_id':user_id,'text':'Рiвень зради: '+str(current_zrada_level)+'\nНизче плінтусу.'}
                                await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)


                            elif int(current_zrada_level) <25:
                                message = {'chat_id':chat_id, 'user_id':user_id,'text':'Рiвень зради: '+str(current_zrada_level)+'\nНизький.'}
                                await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                            elif int(current_zrada_level) <50:
                                message = {'chat_id':chat_id, 'user_id':user_id,'text':'Рiвень зради: '+str(current_zrada_level)+'\nПомiрний.'}
                                await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                #zelensky
                        elif text in zelensky:
                            message = {'chat_id':chat_id, 'user_id':user_id,'text':random.choice(ze_list)}
                            await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                        
                #mc_chicken
                        elif text in mc_chicken:
                            txt = random.randint(1,51)
                            message = {'chat_id':chat_id, 'user_id':user_id,'text':'Эквiвалент у макчiкенах: '+str(txt)}
                            await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                #putin
                        elif text in putin:
                            message = {'chat_id':chat_id, 'user_id':user_id,'text':random.choice(pu_list)}
                            await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                #btc_price
                        
                        elif text in btc_price:
                                async with session.get('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT',timeout=5) as resp:
                                    data =  await resp.json()
                                    symbol = data['symbol']
                                    price = float(data['price'])
                                    price = "{:.2f}".format(price)
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':str(symbol)+': '+str(price)}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                #zrada or peremoga random
                        elif re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text) in zrada_or_peremoga:
                            change_chance = random.randint(1, 11)

                            if change_chance <5:
                                current_zrada_level = int(current_zrada_level)+zrada_change
                                cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                                message = {'chat_id':chat_id, 'user_id':user_id,'text':'Схоже на зраду.\nРiвень зради росте до '+str(current_zrada_level)+'.\nРiвень перемоги впав.','reply_to_message_id':message_id}
                                await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                            elif change_chance >5:
                                current_zrada_level = int(current_zrada_level)-peremoga_change
                                cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                                message = {'chat_id':chat_id, 'user_id':user_id,'text':'Схоже на перемогу!\nРiвень зради впав до '+str(current_zrada_level)+'.\nРiвень перемоги вирiс.','reply_to_message_id':message_id}
                                await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                              
                #zrada change
                        elif re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text) in zrada:

                            if zrada_event == False and peremoga_event == False:
                                print(event_start_chance)
                                if event_start_chance <=20:
                                    
                                    event_start = datetime.datetime.now()
                                    zrada_event = True
                                    current_zrada_level = int(current_zrada_level)+(zrada_change*2)
                                    current_zrada_level = str(current_zrada_level)
                                    cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':'Астрологи оголосили тиждень зради.\nУсі зміни у рівні зради буде подвоєно.\nРiвень зради росте до '+current_zrada_level+'.\nРiвень перемоги впав.\nДякую за увагу'}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                                elif event_start_chance >20:
                                    current_zrada_level = int(current_zrada_level)+zrada_change
                                    current_zrada_level = str(current_zrada_level)
                                    cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':'Рiвень зради росте до '+current_zrada_level+'.\nРiвень перемоги впав.'}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                            elif peremoga_event == True:
                                    current_zrada_level = int(current_zrada_level)+zrada_change
                                    current_zrada_level = str(current_zrada_level)
                                    cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':'Триває тиждень перемоги.\nАле рiвень зради все одно росте до '+current_zrada_level+'.\nРiвень перемоги впав.'}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                            elif zrada_event == True:
                                    current_zrada_level = int(current_zrada_level)+(zrada_change*2)
                                    current_zrada_level = str(current_zrada_level)
                                    cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':'Триває тиждень зради.Рiвень зради росте до '+current_zrada_level+'.\nРiвень перемоги впав.'}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                            else:
                                message = {'chat_id':chat_id, 'user_id':user_id,'text':'Перевiр мій код, строка 165'}
                                await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)


                            
                #peremoga change
                        elif re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text) in peremoga:

                            if zrada_event == False and peremoga_event == False:
                                if event_start_chance <=20:
                                    event_start = datetime.datetime.now()
                                    peremoga_event = True
                                    current_zrada_level = int(current_zrada_level)-(peremoga_change*2)
                                    current_zrada_level = str(current_zrada_level)
                                    cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':'Астрологи оголосили тиждень перемоги.\nУсі зміни у рівні перемоги буде подвоєно.\nРiвень зради падає до '+current_zrada_level+'.\nРiвень перемоги виріс.\nДякую за увагу'}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                                elif event_start_chance >20:
                                    current_zrada_level = int(current_zrada_level)-peremoga_change
                                    current_zrada_level = str(current_zrada_level)
                                    cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':'Рiвень зради впав до '+current_zrada_level+'.\nРiвень перемоги вирiс.'}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                                    
                            elif peremoga_event == True:
                                    current_zrada_level = int(current_zrada_level)-(peremoga_change*2)
                                    current_zrada_level = str(current_zrada_level)
                                    cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':'Триває тиждень перемоги.\nРівень зради падає до '+current_zrada_level+'.\nРiвень перемоги виріс.'}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                            elif zrada_event == True:
                                    current_zrada_level = int(current_zrada_level)-peremoga_change
                                    current_zrada_level = str(current_zrada_level)
                                    cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':'Триває тиждень зради.Але рівень її рівень попри все падає до '+current_zrada_level+'.\nРiвень перемоги виріс.'}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                            else:
                                message = {'chat_id':chat_id, 'user_id':user_id,'text':'Перевiр мій код, строка 195'}
                                await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                            
                #word by word check
                        elif text not in zrada and text not in peremoga and text not in zrada_or_peremoga:
                            words = re.sub(r"[-()#@;:<>{}`+=~|.!?,]", "", text).split()
                            for word in words:
                                if word in zrada:
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':random.choice(zrada_mention_replies),'reply_to_message_id':message_id}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                                elif word in peremoga:
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':random.choice(peremoga_mention_replies),'reply_to_message_id':message_id}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                                elif word in zrada_mention:
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':random.choice(zrada_mention_replies)}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                                elif word in peremoga_mention:
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':random.choice(peremoga_mention_replies),'reply_to_message_id':message_id}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                #bmw
                                elif word in bmw:
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':'Беха топ','reply_to_message_id':message_id}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                #mamka
                                elif word in mamka:
                                    txt = random.choice(mamka_response)
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':txt,'reply_to_message_id':message_id}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                                elif word in random_words:
                                    if event_start_chance <=50:
                                        txt = random.choice(random_replies)
                                        message = {'chat_id':chat_id, 'user_id':user_id,'text':txt,'reply_to_message_id':message_id}
                                        await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                                    elif event_start_chance >50:
                                        txt = random.choice(random_replies)
                                        message = {'chat_id':chat_id, 'user_id':user_id,'text':txt}
                                        await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)



                                else:
                                    message = {'chat_id':my_id, 'user_id':my_id,'text':'Nothing was catched'}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                                    pass
                    
        except Exception as e:
             async with aiohttp.ClientSession() as session:
                chat_id = my_id
                user_id =  my_id
                async with session.get(tel_api+tel_token+'/getUpdates',timeout=5) as resp:
                    data =  await resp.json()
                    message = {'chat_id':chat_id, 'user_id':user_id,'text':str(e)}
                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
            
                print(e)
                pass
        conn.commit()
        


if __name__=='__main__':
     while True:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(bot())
        time.sleep(1)