import requests as r
import json
import asyncio
import aiohttp
import random
loop = asyncio.new_event_loop()
import time
import psycopg2
from  dict import ze_list, zrada, zelensky, zrada_mention, peremoga, peremoga_mention, pu_list, putin, bmw, mamka,mamka_response, status

conn = psycopg2.connect(database="neondb",
host="ep-lucky-sea-840602.eu-central-1.aws.neon.tech",
user="Crabolog",
password="EF6TAl7jwbRu",
port="5432")
chat = '-1002092175489'
tel_token = '6694398809:AAErdp4f0KRoWJ-8F8daRzwmXvl3vJClBo8'
tel_api = 'https://api.telegram.org/bot'

async def bot():
    update = None
    offset = 1
    while True:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM zrada_level WHERE id = 1")
        current_zrada_level = cursor.fetchone()[2]
        change = random.randint(1,21)
        time.sleep(0.5)
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
    
                        #status check
                        if text in status:
                            current_zrada_level = int(current_zrada_level)
                            if int(current_zrada_level) > 150:
                                 message = {'chat_id':chat_id, 'user_id':user_id,'text':'Рiвень зради: '+str(current_zrada_level)+'\nКритично низький рiвень перемоги.'}
                                 await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                            elif current_zrada_level > 50:
                                message = {'chat_id':chat_id, 'user_id':user_id,'text':'Рiвень зради: '+str(current_zrada_level)+'\nВисокий рiвень.'}
                                await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                            elif int(current_zrada_level) <25:
                                 message = {'chat_id':chat_id, 'user_id':user_id,'text':'Рiвень зради: '+str(current_zrada_level)+'\nНизький.'}
                                 await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                            elif int(current_zrada_level) <50:
                                 message = {'chat_id':chat_id, 'user_id':user_id,'text':'Рiвень зради: '+str(current_zrada_level)+'\nПомiрний.'}
                                 await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                        #zelensky
                        elif text in zelensky:
                            txt = random.choice(ze_list)
                            message = {'chat_id':chat_id, 'user_id':user_id,'text':txt}
                            await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

            
                        elif text in zrada:
                            current_zrada_level = int(current_zrada_level)+change
                            print('here')
                            print(current_zrada_level)
                            current_zrada_level = str(current_zrada_level)
                            cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                            message = {'chat_id':chat_id, 'user_id':user_id,'text':'Рiвень зради росте до '+current_zrada_level+'.\nРiвень перемоги впав.'}
                            await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                            

                        elif text in peremoga:
                            current_zrada_level = int(current_zrada_level)-change
                            current_zrada_level = str(current_zrada_level)
                            cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                            message = {'chat_id':chat_id, 'user_id':user_id,'text':'Рiвень зради впав до '+current_zrada_level+'.\nРiвень перемоги вирiс.'}
                            await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                            
                        
                        elif text not in zrada and text not in peremoga:
                            words = text.split()
                            for word in words:
                                if word in zrada:
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':'Менi почулась зрада?','reply_to_message_id':message_id}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                                elif word in peremoga:
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':'Це вже перемога, чи все ще зрада?','reply_to_message_id':message_id}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                                elif word in zrada_mention:
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':'Десь бачу зраду, де вона.'}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                                elif word in peremoga_mention:
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':'i де та перемога...','reply_to_message_id':message_id}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                                elif word in bmw:
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':'Беха топ','reply_to_message_id':message_id}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                                elif word in mamka:
                                    txt = random.choice(mamka_response)
                                    message = {'chat_id':chat_id, 'user_id':user_id,'text':txt,'reply_to_message_id':message_id}
                                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                                else:
                                    pass
            
        except Exception as e:
             async with aiohttp.ClientSession() as session:
                chat_id = '267601623'
                user_id = '267601623'
                async with session.get(tel_api+tel_token+'/getUpdates?offset='+f"{offset}",timeout=5) as resp:
                    data =  await resp.json()
                    message = {'chat_id':chat_id, 'user_id':user_id,'text':e}
                    await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
            
                print(e)
                pass
        conn.commit()
        


if __name__=='__main__':
    asyncio.set_event_loop(loop)
    loop.run_until_complete(bot())
