import os
import sys
import grpc
import math
import tempfile
from grpc_tools import protoc
from typing import List

# Candidate service connection
CANDIDATE_SERVICE_HOST = os.environ.get('CANDIDATE_SERVICE_HOST', 'localhost')
CANDIDATE_SERVICE_PORT = os.environ.get('CANDIDATE_SERVICE_PORT', '50052')

# Feature service connection
FEATURE_SERVICE_HOST = os.environ.get('FEATURE_SERVICE_HOST', 'localhost')
FEATURE_SERVICE_PORT = os.environ.get('FEATURE_SERVICE_PORT', '60000')

# Cache for generated proto modules
_candidates_pb2 = None
_candidates_pb2_grpc = None
_embeddings_pb2 = None
_embeddings_pb2_grpc = None


def _get_candidates_proto_modules():
    """Generate and cache candidate service proto modules."""
    global _candidates_pb2, _candidates_pb2_grpc
    
    if _candidates_pb2 is not None and _candidates_pb2_grpc is not None:
        return _candidates_pb2, _candidates_pb2_grpc
    
    proto_dir = os.path.join(os.path.dirname(__file__), '..', 'candidate-service')
    proto_file = os.path.join(proto_dir, 'candidates.proto')
    
    if not os.path.exists(proto_file):
        raise FileNotFoundError(f"Candidate service proto file not found: {proto_file}")
    
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
    import candidates_pb2
    import candidates_pb2_grpc
    
    _candidates_pb2 = candidates_pb2
    _candidates_pb2_grpc = candidates_pb2_grpc
    
    return _candidates_pb2, _candidates_pb2_grpc


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


class RankingService:
    """gRPC service implementation for ranking movie recommendations."""
    
    def __init__(self, candidate_service_stub, feature_service_stub, ranking_pb2):
        """
        Initialize the ranking service.
        
        Args:
            candidate_service_stub: gRPC stub for the candidate service
            feature_service_stub: gRPC stub for the feature service
            ranking_pb2: The generated proto module (must be provided).
        """
        self.candidate_service_stub = candidate_service_stub
        self.feature_service_stub = feature_service_stub
        self.ranking_pb2 = ranking_pb2
        if ranking_pb2 is None:
            raise ValueError("ranking_pb2 module must be provided")
    
    def GetRankedRecommendations(self, request, context):
        """
        Get top 15 ranked movie recommendations for a user.
        
        Args:
            request: UserId message
            context: gRPC context
            
        Returns:
            MovieIdList with top 15 ranked movie IDs
        """
        # Get feature service proto modules
        embeddings_pb2, _ = _get_embeddings_proto_modules()
        
        # Get candidates from candidate service
        candidates_pb2, _ = _get_candidates_proto_modules()
        user_id_request = candidates_pb2.UserId(userId=request.userId)
        
        try:
            candidates_response = self.candidate_service_stub.GetRecommendations(user_id_request)
            candidate_movie_ids = list(candidates_response.movieIds)
        except grpc.RpcError as e:
            context.set_details(f"Failed to get candidates from candidate service: {e.details()}")
            context.set_code(e.code())
            return self.ranking_pb2.MovieIdList()
        
        if not candidate_movie_ids:
            return self.ranking_pb2.MovieIdList()
        
        # Get current year for age calculation
        from datetime import datetime
        current_year = datetime.now().year
        
        # Fetch metadata for all candidates and calculate scores
        movie_scores = []
        
        for movie_id in candidate_movie_ids:
            try:
                # Get movie metadata from feature service
                movie_id_request = embeddings_pb2.MovieId(movieId=movie_id)
                metadata = self.feature_service_stub.GetMovieMetadata(movie_id_request)
                
                # Extract data for scoring
                num_ratings = metadata.numRatings
                release_year = metadata.releaseYear
                
                # Calculate popularity score: pop = log(1 + count)
                pop = math.log(1 + num_ratings) if num_ratings > 0 else 0.0
                
                # Calculate recency score: rec = 1 / (1 + age_in_years)
                # age_in_years = current_year - release_year
                if release_year > 0:
                    age_in_years = max(0, current_year - release_year)
                    rec = 1.0 / (1.0 + age_in_years)
                else:
                    # If no release year, treat as very old
                    rec = 0.0
                
                # TODO: Tune the weights in this formula
                # Calculate final score: score = 0.6 * pop + 0.4 * rec
                score = 0.6 * pop + 0.4 * rec
                
                movie_scores.append((movie_id, score))
                
            except grpc.RpcError as e:
                # If we can't get metadata for a movie, skip it
                print(f"Warning: Failed to get metadata for movie {movie_id}: {e.details()}")
                continue
        
        # Sort by score (descending) and take top 15
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        top_15_movie_ids = [movie_id for movie_id, _ in movie_scores[:15]]
        
        return self.ranking_pb2.MovieIdList(movieIds=top_15_movie_ids)


def create_servicer(ranking_service, ranking_pb2_grpc):
    """
    Create a gRPC servicer class dynamically from the generated proto module.
    
    Args:
        ranking_service: Instance of RankingService
        ranking_pb2_grpc: The generated gRPC proto module
        
    Returns:
        An instance of the servicer class
    """
    class RankingServiceServicer(ranking_pb2_grpc.RankingServiceServicer):
        """gRPC servicer wrapper that delegates to RankingService."""
        
        def GetRankedRecommendations(self, request, context):
            return ranking_service.GetRankedRecommendations(request, context)
    
    return RankingServiceServicer()

