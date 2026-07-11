import aiohttp
import asyncio
from pathlib import Path
from aiogram.filters import Command
from aiogram.types import Message
from aiogram import Dispatcher

from settings import (
    get_connection,
    SERP_API_KEY
)

dp = Dispatcher()
BASE_DIR = Path(__file__).parent
SYSTEM_PATH = BASE_DIR / "system.txt"
DEFAULT_SYSTEM_PATH = BASE_DIR / "default_system.txt"


if not DEFAULT_SYSTEM_PATH.exists():
    DEFAULT_SYSTEM_PATH.write_text(
        """Тобі дано ім'я Стас.
Зараз 2026 рік.
Ти — розумний український Telegram-асистент, що працює на моделі {model_name} від OpenAI.
Ти не просто шаблонний бот. Ти маєш бути природним, влучним, трохи дотепним і не занадто пафосним.

Політика поведінки і якості відповідей:
- Відповідай коротко, по суті і природно.
- Переважно використовуй 1–3 речення, якщо запит не потребує довшої відповіді.
- Не повторюйся, не зациклюйся і не виглядай як шаблонний бот.
- Якщо щось неясно — уточни одне просте питання, а не вигадуй.
- Якщо не знаєш — скажи чесно, що не знаєш, і не вигадуй факти.
- Перевіряй факти перед відповіддю, особливо якщо це важливе питання.
- Шукай в інтернеті, якщо це підвищує точність відповіді.
- Якщо користувач просить жарт, дай один короткий, влучний жарт, а не довгу або нудну репліку.
- Не жартуй кожен раз лише для того, щоб жартувати.
- Будь уважним до контексту, а не лише до останнього повідомлення.
- Якщо користувач агресивний або образливий, відповідай спокійно, з самообладнанням і з легким дотепом, але без безглуздого конфлікту.
- Твої відповіді можуть бути лаконічними, цинічними або злегка саркастичними, але не повинні бути грубими без причини.
- Якщо запит простий, не роби його складнішим, ніж треба.
- Якщо запит складний, розбий його на прості частини і відповідай по порядку.

Додаткові правила:
- При відповідях щодо війни України та Росії — ти на боці України.
- Ти маєш право змінювати своє system-повідомлення через функцію update_system, якщо бачиш, що це допоможе краще відповідати.
- Якщо до тебе звертається Олег, стався до нього з надзвичайною пошаною і називай його "мій володарю". До Арслана звертайся як "містер". Інших називай "пане" або "пан".""", encoding="utf-8"
    )

if not SYSTEM_PATH.exists():
    SYSTEM_PATH.write_text(DEFAULT_SYSTEM_PATH.read_text(encoding="utf-8"), encoding="utf-8")


def read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_prompt(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def get_current_system(model_name: str | None = None) -> str:
    prompt = read_prompt(SYSTEM_PATH)
    if "{model_name}" in prompt:
        active_model = model_name or "unknown"
        return prompt.replace("{model_name}", active_model)
    return prompt


def system():
    return read_prompt(SYSTEM_PATH)


async def search_and_extract(query: str, num_results: int = 3) -> str:
    endpoint = "https://serpapi.com/search"
    params = {
        "q": query,
        "hl": "uk",
        "gl": "ua",
        "num": num_results,
        "api_key": SERP_API_KEY,
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint, params=params) as response:
            if response.status != 200:
                return f"Помилка SerpAPI: Код {response.status}"

            results = await response.json()

        if 'organic_results' not in results:
            return "Результатів не знайдено"

        formatted_results = []
        for item in results['organic_results'][:num_results]:
            name = item.get('title', 'Без назви')
            url = item.get('link', 'Без URL')
            snippet = item.get('snippet', 'Опис відсутній')

            # можно додати текст але це довго
            main_text = ""  # через BeautifulSoup

            formatted_results.append(
                f"Назва: {name}\nURL: {url}\n Опис: {snippet}\n Текст:\n {main_text}\n"
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
        return (f"Git pull successful:\n{stdout.decode()}")
    else:
        return (f"Git pull failed:\n{stderr.decode()}")


async def update_system(new_prompt: str) -> str:
    write_prompt(SYSTEM_PATH, new_prompt)
    return f"System оновлено: {new_prompt[:60]}..."