import json
import sqlite3
import numpy as np
import logging
from openai import OpenAI
from pathlib import Path
from settings import (
    OPENAI_API_KEY,
    DB_PATH,
    conn,
    get_connection,
)

logging.basicConfig(level=logging.INFO)
# OpenAI embeddings use cosine similarity; no single official cutoff exists.
# Keep duplicate detection stricter than retrieval to avoid over-deduplication.
save_accuracy = 0.82
search_accuracy = 0.38


BASE_DIR = Path(__file__).parent
SYSTEM_PATH = BASE_DIR / "system.txt"
DEFAULT_SYSTEM_PATH = BASE_DIR / "default_system.txt"

conn.row_factory = sqlite3.Row
logging.info("SQLite DB path: %s", DB_PATH)
try:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        embedding TEXT NOT NULL,
        user_id TEXT NOT NULL
    )
    """)
    conn.commit()
    logging.info("Embeddings table ensured")
except Exception as exc:
    logging.exception("Failed to create embeddings table: %s", exc)

client = OpenAI(api_key=OPENAI_API_KEY)


async def get_embeddings_from_db():
    connection = get_connection()
    try:
        logging.info("Reading embeddings from SQLite DB")
        cursor = connection.execute("SELECT text, embedding, user_id FROM embeddings")
        rows = cursor.fetchall()
        logging.info("Loaded %s embeddings from DB", len(rows))
        return [
            (row["text"], np.array(json.loads(row["embedding"])), row["user_id"])
            for row in rows
        ]
    except Exception as exc:
        logging.exception("Failed to read embeddings from DB: %s", exc)
        return []
    finally:
        connection.close()


async def delete_embedding_from_db(embedding_text: str):
    connection = get_connection()
    try:
        logging.info("Deleting embeddings matching: %s", embedding_text)
        cursor = connection.execute(
            "DELETE FROM embeddings WHERE text LIKE ?",
            (f"%{embedding_text}%",)
        )
        connection.commit()
        logging.info("Deleted row count: %s", cursor.rowcount)
        return cursor.rowcount > 0
    except Exception as exc:
        logging.exception("Failed to delete embeddings: %s", exc)
        return False
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
        logging.info("Saving embedding for user %s", user_id)
        existing_embeddings = await get_embeddings_from_db()
        logging.info("Checking %s existing embeddings for similarity", len(existing_embeddings))
        for existing_text, existing_embedding, existing_user_id in existing_embeddings:
            similarity = cosine_similarity(embedding, existing_embedding)
            if similarity >= save_accuracy:
                logging.info("Duplicate embedding found; similarity=%s text=%s", similarity, existing_text)
                return
        embedding_rounded = np.round(embedding, 8)
        embedding_list = embedding_rounded.tolist()
        user_id = str(user_id)
        connection.execute(
            "INSERT INTO embeddings (text, embedding, user_id) VALUES (?, ?, ?)",
            (text, json.dumps(embedding_list), user_id),
        )
        connection.commit()
        logging.info("Inserted embedding into DB: %s", text[:80])
    except Exception as exc:
        logging.exception("Failed to save embedding: %s", exc)
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
    logging.info("Searching for similar messages against %s saved embeddings", len(embeddings_db))
    for saved_text, saved_embedding, saved_user_id in embeddings_db:
        similarity = cosine_similarity(new_embedding, saved_embedding)
        if similarity >= search_accuracy:
            similar_messages.append((saved_text, similarity, saved_user_id))
    logging.info("Found %s similar messages", len(similar_messages))
    return similar_messages
