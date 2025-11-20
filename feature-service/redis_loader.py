import os
import pickle
import redis
import numpy as np

# Redis and Embedding Constants
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
USER_EMBEDDING_PATH = '../embedding-calc/user_embeddings.pkl'
MOVIE_EMBEDDING_PATH = '../embedding-calc/movie_embeddings.pkl'
USER_PREFIX = 'user_embedding:'
MOVIE_PREFIX = 'movie_embedding:'

# Connect to Redis
def get_redis_client():
    """Get a Redis client connection."""
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

# Load embeddings from disk
def load_pkl(path):
    """Load a pickle file from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def ensure_numpy(df):
    """
    Convert DataFrame to numpy arrays.
    Sometimes it's a DataFrame with index, sometimes numpy array.
    """
    if hasattr(df, "index"):
        # DataFrame: index holds the ID
        if df.index.name is None or df.index.name == 'None':
            raise Exception(f'Embeddings DataFrame at {df} has unknown index')
        return df.index.to_numpy(), df.values
    elif isinstance(df, np.ndarray):
        # Just a numpy array, not expected here
        raise Exception('Direct numpy arrays not supported (expect DataFrames with index)')
    else:
        raise Exception('Unknown embedding type ' + str(type(df)))

# Upload all embeddings to Redis
def upload_embeddings(redis_client=None):
    """
    Upload all user and movie embeddings to Redis.
    
    Args:
        redis_client: Optional Redis client. If None, creates a new one.
    """
    if redis_client is None:
        redis_client = get_redis_client()
    
    # Users
    user_df = load_pkl(USER_EMBEDDING_PATH)
    user_ids, user_embs = ensure_numpy(user_df)
    pipe = redis_client.pipeline()
    for uid, emb in zip(user_ids, user_embs):
        # Store as bytes for efficiency, could use pickle or np.tobytes
        pipe.set(f"{USER_PREFIX}{uid}", pickle.dumps(emb.astype(np.float32)))
    pipe.set('user_embedding_ids', pickle.dumps(list(map(int, user_ids))))
    pipe.execute()

    # Movies
    movie_df = load_pkl(MOVIE_EMBEDDING_PATH)
    mov_ids, mov_embs = ensure_numpy(movie_df)
    pipe = redis_client.pipeline()
    for mid, emb in zip(mov_ids, mov_embs):
        pipe.set(f"{MOVIE_PREFIX}{mid}", pickle.dumps(emb.astype(np.float32)))
    pipe.set('movie_embedding_ids', pickle.dumps(list(map(int, mov_ids))))
    pipe.execute()

