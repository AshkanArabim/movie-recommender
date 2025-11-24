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

    def GetMovieEmbeddingsBatch(self, request, context):
        """Get embeddings for multiple movies in a single call for efficiency."""
        embeddings = []
        for movie_id in request.movieIds:
            key = f"{MOVIE_PREFIX}{movie_id}"
            val = self.redis_client.get(key)
            if val is not None:
                emb = pickle.loads(val)
                movie_emb = self.embeddings_pb2.MovieEmbedding(
                    movieId=movie_id,
                    values=list(map(float, emb))
                )
                embeddings.append(movie_emb)
        return self.embeddings_pb2.MovieEmbeddingsBatch(embeddings=embeddings)

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
        self._add_to_watched(request.userId, request.movieId)
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

    def _add_to_watched(self, user_id, movie_id):
        """
        Helper method to add a movie to a user's watched set.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
        """
        key = f"{USER_PREFIX}{user_id}:watched"
        # Use sadd to add to set (idempotent, O(1))
        self.redis_client.sadd(key, movie_id)

    def _update_user_embedding_with_movie(self, user_id, movie_id, weight, context):
        """
        Update user embedding using exponential moving average formula.
        
        Formula: u_new = (1 - η) * u + η * w * m
        where:
        - u is the current user embedding
        - m is the movie embedding
        - w is the weight (1 for like, -0.5 for dislike)
        - η (eta) is 0.2
        
        The result is normalized before storing.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            weight: Weight value (1.0 for like, -0.5 for dislike)
            context: gRPC context for error handling
            
        Returns:
            Empty response or None if error occurred
        """
        eta = 0.2
        
        # Get user embedding
        user_key = f"{USER_PREFIX}{user_id}"
        user_val = self.redis_client.get(user_key)
        if user_val is None:
            # If user doesn't exist, initialize with zero vector
            u = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        else:
            u = pickle.loads(user_val)
            if not isinstance(u, np.ndarray):
                u = np.array(u, dtype=np.float32)
        
        # Get movie embedding
        movie_key = f"{MOVIE_PREFIX}{movie_id}"
        movie_val = self.redis_client.get(movie_key)
        if movie_val is None:
            context.set_details(f"Movie embedding {movie_id} not found")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return self.embeddings_pb2.Empty()
        
        m = pickle.loads(movie_val)
        if not isinstance(m, np.ndarray):
            m = np.array(m, dtype=np.float32)
        
        # Apply exponential moving average formula: u_new = (1 - η) * u + η * w * m
        u_new = (1 - eta) * u + eta * weight * m
        
        # Normalize the embedding (L2 normalization)
        norm = np.linalg.norm(u_new)
        if norm > 0:
            u_new = u_new / norm
        else:
            # If norm is zero (shouldn't happen in practice), keep as is
            pass
        
        # Store normalized embedding back to Redis
        self.redis_client.set(user_key, pickle.dumps(u_new.astype(np.float32)))
        
        # Add movie to user's watched set
        self._add_to_watched(user_id, movie_id)
        
        return self.embeddings_pb2.Empty()

    def LikeMovie(self, request, context):
        """
        Update user embedding based on a like for a movie.
        Uses exponential moving average with weight w=1.0.
        """
        return self._update_user_embedding_with_movie(
            request.userId, 
            request.movieId, 
            weight=1.0, 
            context=context
        )

    def DislikeMovie(self, request, context):
        """
        Update user embedding based on a dislike for a movie.
        Uses exponential moving average with weight w=-0.5.
        """
        return self._update_user_embedding_with_movie(
            request.userId, 
            request.movieId, 
            weight=-0.5, 
            context=context
        )


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
        
        def GetMovieEmbeddingsBatch(self, request, context):
            return embedding_service.GetMovieEmbeddingsBatch(request, context)
        
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
        
        def LikeMovie(self, request, context):
            return embedding_service.LikeMovie(request, context)
        
        def DislikeMovie(self, request, context):
            return embedding_service.DislikeMovie(request, context)
    
    return EmbeddingServiceServicer()

