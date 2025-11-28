import os
import sys
import grpc
import numpy as np
import faiss
import tempfile
from grpc_tools import protoc
from typing import List, Tuple

# GPU detection will be done at runtime using faiss.get_num_gpus()

# Feature service connection
FEATURE_SERVICE_HOST = os.environ.get('FEATURE_SERVICE_HOST', 'localhost')
FEATURE_SERVICE_PORT = os.environ.get('FEATURE_SERVICE_PORT', '60000')
EMBEDDING_DIMENSION = 64

# Cache for generated proto modules
_embeddings_pb2 = None
_embeddings_pb2_grpc = None


def _get_embeddings_proto_modules():
    """Generate and cache feature service proto modules."""
    global _embeddings_pb2, _embeddings_pb2_grpc
    
    if _embeddings_pb2 is not None and _embeddings_pb2_grpc is not None:
        return _embeddings_pb2, _embeddings_pb2_grpc
    
    proto_dir = os.path.join(os.path.dirname(__file__), '..', 'feature-service')
    proto_file = os.path.join(proto_dir, 'embeddings.proto')
    
    if not os.path.exists(proto_file):
        raise FileNotFoundError(f"Feature service proto file not found: {proto_file}")
    
    out_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    try:
        os.chdir(proto_dir)
        protoc.main((
            '',
            f'-I{proto_dir}',
            f'--python_out={out_dir}',
            f'--grpc_python_out={out_dir}',
            os.path.basename(proto_file),
        ))
    finally:
        os.chdir(original_cwd)
    
    sys.path.insert(0, out_dir)
    import embeddings_pb2
    import embeddings_pb2_grpc
    
    _embeddings_pb2 = embeddings_pb2
    _embeddings_pb2_grpc = embeddings_pb2_grpc
    
    return _embeddings_pb2, _embeddings_pb2_grpc


class CandidateService:
    """gRPC service implementation for movie candidate recommendations."""
    
    def __init__(self, feature_service_stub, candidates_pb2):
        """
        Initialize the candidate service.
        
        Args:
            feature_service_stub: gRPC stub for the feature service
            candidates_pb2: The generated proto module (must be provided).
        """
        self.feature_service_stub = feature_service_stub
        self.candidates_pb2 = candidates_pb2
        if candidates_pb2 is None:
            raise ValueError("candidates_pb2 module must be provided")
        
        # FAISS index with ID mapping
        self.index = None
        
        # Load movie embeddings and build FAISS index on startup
        self._load_movie_embeddings()
    
    def _load_movie_embeddings(self):
        """Load all movie embeddings from feature service and build FAISS index."""
        print("Loading movie embeddings from feature service...")
        
        # Get feature service proto modules
        embeddings_pb2, embeddings_pb2_grpc = _get_embeddings_proto_modules()
        
        # Get all movie IDs
        empty = embeddings_pb2.Empty()
        movie_id_list = self.feature_service_stub.ListMovieIds(empty)
        movie_ids = list(movie_id_list.movieIds)
        
        if not movie_ids:
            raise ValueError("No movie IDs found in feature service")
        
        print(f"Found {len(movie_ids)} movies. Loading embeddings...")
        
        # Load embeddings in batches (to avoid gRPC message size limits)
        batch_size = 1000
        all_embeddings = []
        valid_movie_ids = []
        
        for i in range(0, len(movie_ids), batch_size):
            batch_ids = movie_ids[i:i + batch_size]
            movie_id_list_request = embeddings_pb2.MovieIdList(movieIds=batch_ids)
            batch_response = self.feature_service_stub.GetMovieEmbeddingsBatch(movie_id_list_request)
            
            for movie_emb in batch_response.embeddings:
                valid_movie_ids.append(movie_emb.movieId)
                all_embeddings.append(np.array(movie_emb.values, dtype=np.float32))
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Loaded {min(i + batch_size, len(movie_ids))} / {len(movie_ids)} movies...")
        
        if not all_embeddings:
            raise ValueError("No movie embeddings loaded")
        
        # Convert to numpy array
        embedding_matrix = np.vstack(all_embeddings).astype(np.float32)
        movie_ids_array = np.array(valid_movie_ids, dtype=np.int64)
        
        print(f"Loaded {len(all_embeddings)} movie embeddings. Building FAISS index...")
        
        # Normalize embeddings for cosine similarity (using inner product)
        faiss.normalize_L2(embedding_matrix)
        
        # Create FAISS index - use GPU if available, otherwise CPU
        dimension = embedding_matrix.shape[1]
        
        # Create base index
        base_index = faiss.IndexFlatIP(dimension)
        
        try:
            num_gpus = faiss.get_num_gpus()
            if num_gpus > 0:
                print(f"Using GPU FAISS (found {num_gpus} GPU(s))")
                # Use GPU resource
                res = faiss.StandardGpuResources()
                # Move base index to GPU
                base_index = faiss.index_cpu_to_gpu(res, 0, base_index)
            else:
                print("Using CPU FAISS (no GPUs detected)")
        except Exception as e:
            # Fallback to CPU if GPU setup fails
            print(f"GPU setup failed ({e}), falling back to CPU FAISS")
        
        # Wrap with IndexIDMap to enable ID-based operations
        self.index = faiss.IndexIDMap(base_index)
        
        # Add embeddings with movie IDs
        self.index.add_with_ids(embedding_matrix, movie_ids_array)
        
        print(f"FAISS index built successfully with {self.index.ntotal} vectors")
    
    def GetRecommendations(self, request, context):
        """
        Get top 50 movie recommendations for a user.
        
        Args:
            request: UserId message
            context: gRPC context
            
        Returns:
            MovieIdList with top 50 recommended movie IDs
        """
        if self.index is None:
            context.set_details("Movie embeddings not loaded. Service may still be initializing.")
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            return self.candidates_pb2.MovieIdList()
        
        # Get feature service proto modules
        embeddings_pb2, _ = _get_embeddings_proto_modules()
        
        # Get user embedding from feature service
        user_id_request = embeddings_pb2.UserId(userId=request.userId)
        try:
            user_embedding_reply = self.feature_service_stub.GetUserEmbedding(user_id_request)
            user_embedding = np.array(user_embedding_reply.values, dtype=np.float32)
        except grpc.RpcError as e:
            context.set_details(f"Failed to get user embedding: {e.details()}")
            context.set_code(e.code())
            return self.candidates_pb2.MovieIdList()
        
        # Get watched movies from feature service
        try:
            watched_movies_reply = self.feature_service_stub.GetWatchedMovies(user_id_request)
            watched_movie_ids = np.array(watched_movies_reply.movieIds, dtype=np.int64)
        except grpc.RpcError as e:
            # If we can't get watched movies, proceed without filtering (better than failing)
            print(f"Warning: Failed to get watched movies for user {request.userId}: {e.details()}")
            watched_movie_ids = np.array([], dtype=np.int64)
        
        # Normalize user embedding
        user_embedding = user_embedding.reshape(1, -1)
        faiss.normalize_L2(user_embedding)
        
        # Search for top 50 movies, excluding watched ones using IDSelectorNot
        k = min(50, self.index.ntotal)
        
        if len(watched_movie_ids) > 0:
            # Create ID selector to exclude watched movies
            selector = faiss.IDSelectorNot(watched_movie_ids)
            distances, indices = self.index.search_with_idselector(user_embedding, k, selector)
        else:
            # No watched movies to exclude, do regular search
            distances, indices = self.index.search(user_embedding, k)
        
        # Convert indices (which are now movie IDs) to list
        recommended_movie_ids = [int(movie_id) for movie_id in indices[0] if movie_id >= 0]
        
        return self.candidates_pb2.MovieIdList(movieIds=recommended_movie_ids)


def create_servicer(candidate_service, candidates_pb2_grpc):
    """
    Create a gRPC servicer class dynamically from the generated proto module.
    
    Args:
        candidate_service: Instance of CandidateService
        candidates_pb2_grpc: The generated gRPC proto module
        
    Returns:
        An instance of the servicer class
    """
    class CandidateServiceServicer(candidates_pb2_grpc.CandidateServiceServicer):
        """gRPC servicer wrapper that delegates to CandidateService."""
        
        def GetRecommendations(self, request, context):
            return candidate_service.GetRecommendations(request, context)
    
    return CandidateServiceServicer()

