import pickle
import grpc
import numpy as np
import redis_loader
from redis_loader import get_redis_client, USER_PREFIX, MOVIE_PREFIX, EMBEDDING_DIMENSION


class EmbeddingService:
    """gRPC service implementation for embedding operations."""
    
    def __init__(self, redis_client=None, embeddings_pb2=None):
        """
        Initialize the embedding service.
        
        Args:
            redis_client: Optional Redis client. If None, creates a new one.
            embeddings_pb2: The generated proto module (must be provided).
        """
        self.redis_client = redis_client or get_redis_client()
        self.embeddings_pb2 = embeddings_pb2
        if embeddings_pb2 is None:
            raise ValueError("embeddings_pb2 module must be provided")
    
    def GetUserEmbedding(self, request, context):
        """Get embedding for a specific user. Returns zero vector if not found."""
        key = f"{USER_PREFIX}{request.userId}"
        val = self.redis_client.get(key)
        if val is None:
            # Return zero vector if user not found
            zero_emb = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
            return self.embeddings_pb2.EmbeddingReply(values=list(map(float, zero_emb)))
        emb = pickle.loads(val)
        return self.embeddings_pb2.EmbeddingReply(values=list(map(float, emb)))

    def GetMovieEmbedding(self, request, context):
        """Get embedding for a specific movie."""
        key = f"{MOVIE_PREFIX}{request.movieId}"
        val = self.redis_client.get(key)
        if val is None:
            context.set_details(f"Movie embedding {request.movieId} not found")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return self.embeddings_pb2.EmbeddingReply()
        emb = pickle.loads(val)
        return self.embeddings_pb2.EmbeddingReply(values=list(map(float, emb)))

    def ListUserIds(self, request, context):
        """List all available user IDs."""
        ids_blob = self.redis_client.get('user_ids')
        if ids_blob is None:
            return self.embeddings_pb2.UserIdList(userIds=[])
        user_ids = pickle.loads(ids_blob)
        return self.embeddings_pb2.UserIdList(userIds=list(map(int, user_ids)))

    def ListMovieIds(self, request, context):
        """List all available movie IDs."""
        ids_blob = self.redis_client.get('movie_ids')
        if ids_blob is None:
            return self.embeddings_pb2.MovieIdList(movieIds=[])
        movie_ids = pickle.loads(ids_blob)
        return self.embeddings_pb2.MovieIdList(movieIds=list(map(int, movie_ids)))

    def SetUserEmbedding(self, request, context):
        """Set the user embedding."""
        if not request.values:
            context.set_details("Embedding values cannot be empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return self.embeddings_pb2.Empty()
        
        # Validate dimension
        if len(request.values) != EMBEDDING_DIMENSION:
            context.set_details(f"Embedding dimension must be {EMBEDDING_DIMENSION}, got {len(request.values)}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return self.embeddings_pb2.Empty()
        
        # Convert to numpy array and store
        emb = np.array(request.values, dtype=np.float32)
        key = f"{USER_PREFIX}{request.userId}"
        self.redis_client.set(key, pickle.dumps(emb))
        return self.embeddings_pb2.Empty()

    def AddWatchedMovie(self, request, context):
        """Add a movie to a user's watched set."""
        key = f"{USER_PREFIX}{request.userId}:watched"
        # Use sadd to add to set (idempotent, O(1))
        self.redis_client.sadd(key, request.movieId)
        return self.embeddings_pb2.Empty()

    def RemoveWatchedMovie(self, request, context):
        """Remove a movie from a user's watched set."""
        key = f"{USER_PREFIX}{request.userId}:watched"
        # Use srem to remove from set (O(1))
        self.redis_client.srem(key, request.movieId)
        return self.embeddings_pb2.Empty()

    def HasWatchedMovie(self, request, context):
        """Check if a user has watched a movie. O(1) operation using set membership."""
        key = f"{USER_PREFIX}{request.userId}:watched"
        # Use sismember to check membership (O(1))
        has_watched = self.redis_client.sismember(key, request.movieId)
        return self.embeddings_pb2.HasWatchedReply(hasWatched=bool(has_watched))


def create_servicer(embedding_service, embeddings_pb2_grpc):
    """
    Create a gRPC servicer class dynamically from the generated proto module.
    
    Args:
        embedding_service: Instance of EmbeddingService
        embeddings_pb2_grpc: The generated gRPC proto module
        
    Returns:
        An instance of the servicer class
    """
    class EmbeddingServiceServicer(embeddings_pb2_grpc.EmbeddingServiceServicer):
        """gRPC servicer wrapper that delegates to EmbeddingService."""
        
        def GetUserEmbedding(self, request, context):
            return embedding_service.GetUserEmbedding(request, context)
        
        def SetUserEmbedding(self, request, context):
            return embedding_service.SetUserEmbedding(request, context)
        
        def GetMovieEmbedding(self, request, context):
            return embedding_service.GetMovieEmbedding(request, context)
        
        def ListUserIds(self, request, context):
            return embedding_service.ListUserIds(request, context)
        
        def ListMovieIds(self, request, context):
            return embedding_service.ListMovieIds(request, context)
        
        def AddWatchedMovie(self, request, context):
            return embedding_service.AddWatchedMovie(request, context)
        
        def RemoveWatchedMovie(self, request, context):
            return embedding_service.RemoveWatchedMovie(request, context)
        
        def HasWatchedMovie(self, request, context):
            return embedding_service.HasWatchedMovie(request, context)
    
    return EmbeddingServiceServicer()

