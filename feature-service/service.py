import pickle
import grpc
import redis_loader
from redis_loader import get_redis_client, USER_PREFIX, MOVIE_PREFIX


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
        """Get embedding for a specific user."""
        key = f"{USER_PREFIX}{request.userId}"
        val = self.redis_client.get(key)
        if val is None:
            context.set_details(f"User embedding {request.userId} not found")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return self.embeddings_pb2.EmbeddingReply()
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
        ids_blob = self.redis_client.get('user_embedding_ids')
        if ids_blob is None:
            return self.embeddings_pb2.UserIdList(userIds=[])
        user_ids = pickle.loads(ids_blob)
        return self.embeddings_pb2.UserIdList(userIds=list(map(int, user_ids)))

    def ListMovieIds(self, request, context):
        """List all available movie IDs."""
        ids_blob = self.redis_client.get('movie_embedding_ids')
        if ids_blob is None:
            return self.embeddings_pb2.MovieIdList(movieIds=[])
        movie_ids = pickle.loads(ids_blob)
        return self.embeddings_pb2.MovieIdList(movieIds=list(map(int, movie_ids)))


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
        
        def GetMovieEmbedding(self, request, context):
            return embedding_service.GetMovieEmbedding(request, context)
        
        def ListUserIds(self, request, context):
            return embedding_service.ListUserIds(request, context)
        
        def ListMovieIds(self, request, context):
            return embedding_service.ListMovieIds(request, context)
    
    return EmbeddingServiceServicer()

