import requests as r
import json
import asyncio
import aiohttp
import random
import time
import re
import psycopg2
from  dict import *
from settings import *
loop = asyncio.new_event_loop()


async def bot():
    update = None
    offset = 1
    while True:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM zrada_level WHERE id = 1")
        current_zrada_level = cursor.fetchone()[2]
        change = random.randint(1,21)
        time.sleep(0.7)
        try:
            
            async with aiohttp.ClientSession() as session:
                async with session.get(tel_api+tel_token+'/getUpdates?offset='+f"{offset}",timeout=5) as resp:
                    data =  await resp.json()
                    update_id = data['result'][-1]['update_id']
                    offset = data['result'][-1]['update_id']

                    if update_id == update:
                        time.sleep(1.3)
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


                        elif re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text) in zrada_or_peremoga:
                            chance = random.randint(1, 11)

                            if chance <5:
                                current_zrada_level = int(current_zrada_level)+change
                                current_zrada_level = str(current_zrada_level)
                                cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                                message = {'chat_id':chat_id, 'user_id':user_id,'text':'Схоже на зраду.\nРiвень зради росте до '+current_zrada_level+'.\nРiвень перемоги впав.','reply_to_message_id':message_id}
                                await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)

                            elif chance >5:
                                current_zrada_level = int(current_zrada_level)-change
                                current_zrada_level = str(current_zrada_level)
                                cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                                message = {'chat_id':chat_id, 'user_id':user_id,'text':'Схоже на перемогу!\nРiвень зради впав до '+current_zrada_level+'.\nРiвень перемоги вирiс.','reply_to_message_id':message_id}
                                await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                            
                                

            
                        elif re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text) in zrada:
                            current_zrada_level = int(current_zrada_level)+change
                            current_zrada_level = str(current_zrada_level)
                            cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                            message = {'chat_id':chat_id, 'user_id':user_id,'text':'Рiвень зради росте до '+current_zrada_level+'.\nРiвень перемоги впав.'}
                            await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                            

                        elif re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text) in peremoga:
                            current_zrada_level = int(current_zrada_level)-change
                            current_zrada_level = str(current_zrada_level)
                            cursor.execute("UPDATE zrada_level set value = "+current_zrada_level+" WHERE id = 1")
                            message = {'chat_id':chat_id, 'user_id':user_id,'text':'Рiвень зради впав до '+current_zrada_level+'.\nРiвень перемоги вирiс.'}
                            await session.post(tel_api+tel_token+'/sendMessage',data=message,timeout=5)
                            
                        
                        elif text not in zrada and text not in peremoga and text not in zrada_or_peremoga:
                            words = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text).split()
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
                chat_id = my_id
                user_id =  my_id
                async with session.get(tel_api+tel_token+'/getUpdates?offset='+f"{offset}",timeout=5) as resp:
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
        time.sleep(5)
    
    
