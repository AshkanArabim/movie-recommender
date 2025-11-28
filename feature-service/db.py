import os
import pickle
import psycopg2
import psycopg2.extras
import numpy as np
from typing import Optional, List, Tuple, Set


# Database connection parameters
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = int(os.environ.get('DB_PORT', 5432))
DB_NAME = os.environ.get('DB_NAME', 'feature_service')
DB_USER = os.environ.get('DB_USER', 'postgres')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'postgres')

EMBEDDING_DIMENSION = 64


def get_db_connection():
    """Get a PostgreSQL database connection."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


def init_schema():
    """Initialize the database schema. Creates tables if they don't exist."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Create user_embeddings table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_embeddings (
                    user_id INTEGER PRIMARY KEY,
                    embedding BYTEA NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create movie_embeddings table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS movie_embeddings (
                    movie_id INTEGER PRIMARY KEY,
                    embedding BYTEA NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create user_watched_movies table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_watched_movies (
                    user_id INTEGER NOT NULL,
                    movie_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, movie_id)
                )
            """)
            
            # Create movie_metadata table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS movie_metadata (
                    movie_id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    release_year INTEGER,
                    genres TEXT[],
                    num_ratings INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better query performance
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_watched_movies_user_id 
                ON user_watched_movies(user_id)
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_watched_movies_movie_id 
                ON user_watched_movies(movie_id)
            """)
            
            conn.commit()
    finally:
        conn.close()


# User Embedding Operations
def get_user_embedding(user_id: int) -> Optional[np.ndarray]:
    """Get user embedding from database. Returns None if not found."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT embedding FROM user_embeddings WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return pickle.loads(row[0])
    finally:
        conn.close()


def set_user_embedding(user_id: int, embedding: np.ndarray):
    """Set user embedding in database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            embedding_bytes = pickle.dumps(embedding.astype(np.float32))
            cur.execute("""
                INSERT INTO user_embeddings (user_id, embedding, updated_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id) 
                DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = CURRENT_TIMESTAMP
            """, (user_id, embedding_bytes))
            conn.commit()
    finally:
        conn.close()


def get_all_user_ids() -> List[int]:
    """Get all user IDs from database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM user_embeddings ORDER BY user_id")
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


# Movie Embedding Operations
def get_movie_embedding(movie_id: int) -> Optional[np.ndarray]:
    """Get movie embedding from database. Returns None if not found."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT embedding FROM movie_embeddings WHERE movie_id = %s", (movie_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return pickle.loads(row[0])
    finally:
        conn.close()


def get_movie_embeddings_batch(movie_ids: List[int]) -> dict:
    """Get multiple movie embeddings from database. Returns dict mapping movie_id to embedding."""
    if not movie_ids:
        return {}
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Use psycopg2's array adapter for safe parameterized query
            cur.execute("""
                SELECT movie_id, embedding 
                FROM movie_embeddings 
                WHERE movie_id = ANY(%s::int[])
            """, (movie_ids,))
            results = {}
            for row in cur.fetchall():
                results[row[0]] = pickle.loads(row[1])
            return results
    finally:
        conn.close()


def set_movie_embedding(movie_id: int, embedding: np.ndarray):
    """Set movie embedding in database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            embedding_bytes = pickle.dumps(embedding.astype(np.float32))
            cur.execute("""
                INSERT INTO movie_embeddings (movie_id, embedding, updated_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (movie_id) 
                DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = CURRENT_TIMESTAMP
            """, (movie_id, embedding_bytes))
            conn.commit()
    finally:
        conn.close()


def get_all_movie_ids() -> List[int]:
    """Get all movie IDs from database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT movie_id FROM movie_embeddings ORDER BY movie_id")
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


# Watched Movies Operations
def add_watched_movie(user_id: int, movie_id: int):
    """Add a movie to a user's watched set in database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_watched_movies (user_id, movie_id, created_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, movie_id) DO NOTHING
            """, (user_id, movie_id))
            conn.commit()
    finally:
        conn.close()


def remove_watched_movie(user_id: int, movie_id: int):
    """Remove a movie from a user's watched set in database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM user_watched_movies 
                WHERE user_id = %s AND movie_id = %s
            """, (user_id, movie_id))
            conn.commit()
    finally:
        conn.close()


def has_watched_movie(user_id: int, movie_id: int) -> bool:
    """Check if a user has watched a movie in database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 1 FROM user_watched_movies 
                WHERE user_id = %s AND movie_id = %s
            """, (user_id, movie_id))
            return cur.fetchone() is not None
    finally:
        conn.close()


def get_watched_movies(user_id: int) -> Set[int]:
    """Get all movies that a user has watched from database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT movie_id FROM user_watched_movies 
                WHERE user_id = %s
            """, (user_id,))
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


# Bulk operations for pre-warming
def load_all_user_embeddings() -> List[Tuple[int, np.ndarray]]:
    """Load all user embeddings from database for pre-warming cache."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id, embedding FROM user_embeddings")
            return [(row[0], pickle.loads(row[1])) for row in cur.fetchall()]
    finally:
        conn.close()


def load_all_movie_embeddings() -> List[Tuple[int, np.ndarray]]:
    """Load all movie embeddings from database for pre-warming cache."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT movie_id, embedding FROM movie_embeddings")
            return [(row[0], pickle.loads(row[1])) for row in cur.fetchall()]
    finally:
        conn.close()


def load_all_watched_movies() -> List[Tuple[int, int]]:
    """Load all watched movie relationships from database for pre-warming cache."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id, movie_id FROM user_watched_movies")
            return [(row[0], row[1]) for row in cur.fetchall()]
    finally:
        conn.close()


# Movie Metadata Operations
def get_movie_metadata(movie_id: int) -> Optional[dict]:
    """Get movie metadata from database. Returns None if not found."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT title, release_year, genres, num_ratings 
                FROM movie_metadata 
                WHERE movie_id = %s
            """, (movie_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return {
                'title': row[0],
                'release_year': row[1],
                'genres': row[2] if row[2] else [],
                'num_ratings': row[3] if row[3] else 0
            }
    finally:
        conn.close()


def set_movie_metadata(movie_id: int, title: str, release_year: Optional[int], 
                      genres: List[str], num_ratings: int = 0):
    """Set movie metadata in database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO movie_metadata (movie_id, title, release_year, genres, num_ratings, created_at)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (movie_id) 
                DO UPDATE SET 
                    title = EXCLUDED.title,
                    release_year = EXCLUDED.release_year,
                    genres = EXCLUDED.genres,
                    num_ratings = EXCLUDED.num_ratings
            """, (movie_id, title, release_year, genres, num_ratings))
            conn.commit()
    finally:
        conn.close()


def has_movie_metadata() -> bool:
    """Check if database has any movie metadata."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM movie_metadata")
            count = cur.fetchone()[0]
            return count > 0
    finally:
        conn.close()

