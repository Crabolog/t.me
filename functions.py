import numpy as np
import logging
from openai import OpenAI
from pathlib import Path
from collections import deque
from aiogram import Bot, Dispatcher, Router
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

logging.basicConfig(level=logging.INFO)
save_accuracy = 0.65
search_accuracy = 0.33


BASE_DIR = Path(__file__).parent
SYSTEM_PATH = BASE_DIR / "system.txt"
DEFAULT_SYSTEM_PATH = BASE_DIR / "default_system.txt"

dp = Dispatcher()
router = Router()
dp.include_router(router)

conn.autocommit = True
cursor = conn.cursor()

client = OpenAI(api_key=OPENAI_API_KEY)


async def get_embeddings_from_db():
    conn = await get_connection()
    query = "SELECT text, embedding, user_id FROM embeddings"
    rows = await conn.fetch(query)
    return [(row['text'], np.array(row['embedding']), row['user_id']) for row in rows]


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
            print('similar vector found')
            print('threshold: ' + str(save_accuracy))
            print('Similarity: ' + str(similarity))
            print('message text: ' + str(existing_text))
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


async def save_embedding(text: str, embedding, user_id: int):
    await save_embedding_to_db(text, embedding, user_id)


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
