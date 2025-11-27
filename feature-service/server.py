import os
import sys
import tempfile
import pickle
from concurrent import futures

import grpc
from grpc_tools import protoc
import numpy as np

import redis_loader
import db
from service import EmbeddingService
from redis_loader import USER_PREFIX, MOVIE_PREFIX


def generate_proto_modules():
    """
    Generate Python modules from the proto file.
    Returns the generated modules (embeddings_pb2, embeddings_pb2_grpc).
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proto_file = os.path.join(script_dir, 'embeddings.proto')
    proto_dir = script_dir
    
    if not os.path.exists(proto_file):
        raise FileNotFoundError(f"Proto file not found: {proto_file}")
    
    out_dir = tempfile.mkdtemp()
    
    # Change to proto directory for protoc to work correctly
    original_cwd = os.getcwd()
    try:
        os.chdir(proto_dir)
        # Generate Python code from proto
        protoc.main((
            '',
            f'-I{proto_dir}',
            f'--python_out={out_dir}',
            f'--grpc_python_out={out_dir}',
            os.path.basename(proto_file),
        ))
    finally:
        os.chdir(original_cwd)
    
    # Add output directory to path
    sys.path.insert(0, out_dir)
    
    # Import generated modules
    import embeddings_pb2
    import embeddings_pb2_grpc
    
    return embeddings_pb2, embeddings_pb2_grpc


def prewarm_cache_from_db(redis_client):
    """Pre-warm Redis cache from PostgreSQL database."""
    print("Pre-warming cache from database...")
    
    # Load user embeddings
    user_embeddings = db.load_all_user_embeddings()
    if user_embeddings:
        pipe = redis_client.pipeline()
        user_ids = []
        for user_id, emb in user_embeddings:
            key = f"{USER_PREFIX}{user_id}"
            pipe.set(key, pickle.dumps(emb.astype(np.float32)))
            user_ids.append(user_id)
        if user_ids:
            pipe.set('user_ids', pickle.dumps(user_ids))
        pipe.execute()
        print(f"Loaded {len(user_embeddings)} user embeddings into cache")
    
    # Load movie embeddings
    movie_embeddings = db.load_all_movie_embeddings()
    if movie_embeddings:
        pipe = redis_client.pipeline()
        movie_ids = []
        for movie_id, emb in movie_embeddings:
            key = f"{MOVIE_PREFIX}{movie_id}"
            pipe.set(key, pickle.dumps(emb.astype(np.float32)))
            movie_ids.append(movie_id)
        if movie_ids:
            pipe.set('movie_ids', pickle.dumps(movie_ids))
        pipe.execute()
        print(f"Loaded {len(movie_embeddings)} movie embeddings into cache")
    
    # Load watched movies
    watched_movies = db.load_all_watched_movies()
    if watched_movies:
        # Group by user_id for efficient Redis operations
        user_watched_map = {}
        for user_id, movie_id in watched_movies:
            if user_id not in user_watched_map:
                user_watched_map[user_id] = []
            user_watched_map[user_id].append(movie_id)
        
        pipe = redis_client.pipeline()
        for user_id, movie_ids in user_watched_map.items():
            key = f"{USER_PREFIX}{user_id}:watched"
            pipe.sadd(key, *movie_ids)
        pipe.execute()
        print(f"Loaded {len(watched_movies)} watched movie relationships into cache")
    
    print("Cache pre-warming completed.")


def initialize_database_from_pickle():
    """Initialize database from pickle files if database is empty."""
    print("Checking if database needs initialization from pickle files...")
    
    # Check if database has any data
    user_ids = db.get_all_user_ids()
    movie_ids = db.get_all_movie_ids()
    
    if not user_ids and not movie_ids:
        print("Database is empty. Loading from pickle files...")
        
        # Load from pickle files
        user_df = redis_loader.load_pkl(redis_loader.USER_EMBEDDING_PATH)
        user_ids_list, user_embs = redis_loader.ensure_numpy(user_df)
        
        movie_df = redis_loader.load_pkl(redis_loader.MOVIE_EMBEDDING_PATH)
        movie_ids_list, movie_embs = redis_loader.ensure_numpy(movie_df)
        
        # Store in database
        print(f"Storing {len(user_ids_list)} user embeddings in database...")
        for uid, emb in zip(user_ids_list, user_embs):
            db.set_user_embedding(int(uid), emb.astype(np.float32))
        
        print(f"Storing {len(movie_ids_list)} movie embeddings in database...")
        for mid, emb in zip(movie_ids_list, movie_embs):
            db.set_movie_embedding(int(mid), emb.astype(np.float32))
        
        print("Database initialized from pickle files.")
    else:
        print(f"Database already contains {len(user_ids)} users and {len(movie_ids)} movies.")


def serve():
    """Start the gRPC server."""
    # Generate proto modules
    embeddings_pb2, embeddings_pb2_grpc = generate_proto_modules()
    
    # Initialize database schema
    print("Initializing database schema...")
    db.init_schema()
    print("Database schema initialized.")
    
    # Initialize database from pickle files if empty
    initialize_database_from_pickle()
    
    # Pre-warm cache from database
    redis_client = redis_loader.get_redis_client()
    prewarm_cache_from_db(redis_client)
    
    # Create service instance
    embedding_service = EmbeddingService(redis_client, embeddings_pb2)
    from service import create_servicer
    servicer = create_servicer(embedding_service, embeddings_pb2_grpc)
    
    # Start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    embeddings_pb2_grpc.add_EmbeddingServiceServicer_to_server(servicer, server)
    grpc_port = os.environ.get('GRPC_PORT', '60000') # TODO: revert to 50051 in docker
    server.add_insecure_port(f'[::]:{grpc_port}')
    print(f"gRPC EmbeddingService serving on port {grpc_port}")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()

