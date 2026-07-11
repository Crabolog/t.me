import json
import sqlite3
import numpy as np
import logging
from openai import OpenAI
from pathlib import Path
from settings import (
    OPENAI_API_KEY,
    conn,
    get_connection,
)

logging.basicConfig(level=logging.INFO)
save_accuracy = 0.65
search_accuracy = 0.33


BASE_DIR = Path(__file__).parent
SYSTEM_PATH = BASE_DIR / "system.txt"
DEFAULT_SYSTEM_PATH = BASE_DIR / "default_system.txt"

conn.row_factory = sqlite3.Row
conn.execute("""
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    embedding TEXT NOT NULL,
    user_id TEXT NOT NULL
)
""")
conn.commit()

client = OpenAI(api_key=OPENAI_API_KEY)


async def get_embeddings_from_db():
    connection = get_connection()
    try:
        cursor = connection.execute("SELECT text, embedding, user_id FROM embeddings")
        rows = cursor.fetchall()
        return [
            (row["text"], np.array(json.loads(row["embedding"])), row["user_id"])
            for row in rows
        ]
    finally:
        connection.close()


async def delete_embedding_from_db(embedding_text: str):
    connection = get_connection()
    try:
        cursor = connection.execute(
            "DELETE FROM embeddings WHERE text LIKE ?",
            (f"%{embedding_text}%",)
        )
        connection.commit()
        return cursor.rowcount > 0
    finally:
        connection.close()


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
    connection = get_connection()
    try:
        existing_embeddings = await get_embeddings_from_db()
        for existing_text, existing_embedding, existing_user_id in existing_embeddings:
            similarity = cosine_similarity(embedding, existing_embedding)
            if similarity >= save_accuracy:
                print('similar vector found')
                print('threshold: ' + str(save_accuracy))
                print('Similarity: ' + str(similarity))
                print('message text: ' + str(existing_text))
                return
        embedding_rounded = np.round(embedding, 8)
        embedding_list = embedding_rounded.tolist()
        user_id = str(user_id)
        connection.execute(
            "INSERT INTO embeddings (text, embedding, user_id) VALUES (?, ?, ?)",
            (text, json.dumps(embedding_list), user_id),
        )
        connection.commit()
    except Exception as e:
        logging.error(e)
    finally:
        connection.close()


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
