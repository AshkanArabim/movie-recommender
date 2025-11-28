import os
import sys
import tempfile
import grpc
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from grpc_tools import protoc

app = FastAPI(title="Movie Recommendation Gateway")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service connection settings
RANKING_SERVICE_HOST = os.environ.get('RANKING_SERVICE_HOST', 'localhost')
RANKING_SERVICE_PORT = os.environ.get('RANKING_SERVICE_PORT', '50053')
FEATURE_SERVICE_HOST = os.environ.get('FEATURE_SERVICE_HOST', 'localhost')
FEATURE_SERVICE_PORT = os.environ.get('FEATURE_SERVICE_PORT', '50051')

# Cache for generated proto modules
_ranking_pb2 = None
_ranking_pb2_grpc = None
_embeddings_pb2 = None
_embeddings_pb2_grpc = None


def _get_ranking_proto_modules():
    """Generate and cache ranking service proto modules."""
    global _ranking_pb2, _ranking_pb2_grpc
    
    if _ranking_pb2 is not None and _ranking_pb2_grpc is not None:
        return _ranking_pb2, _ranking_pb2_grpc
    
    proto_dir = os.path.join(os.path.dirname(__file__), '..', 'ranking-service')
    proto_file = os.path.join(proto_dir, 'ranking.proto')
    
    if not os.path.exists(proto_file):
        raise FileNotFoundError(f"Ranking service proto file not found: {proto_file}")
    
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
    import ranking_pb2
    import ranking_pb2_grpc
    
    _ranking_pb2 = ranking_pb2
    _ranking_pb2_grpc = ranking_pb2_grpc
    
    return _ranking_pb2, _ranking_pb2_grpc


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


# Initialize gRPC clients
ranking_pb2, ranking_pb2_grpc = _get_ranking_proto_modules()
embeddings_pb2, embeddings_pb2_grpc = _get_embeddings_proto_modules()

# Create gRPC channels
ranking_channel = grpc.insecure_channel(f'{RANKING_SERVICE_HOST}:{RANKING_SERVICE_PORT}')
feature_channel = grpc.insecure_channel(f'{FEATURE_SERVICE_HOST}:{FEATURE_SERVICE_PORT}')

# Create stubs
ranking_stub = ranking_pb2_grpc.RankingServiceStub(ranking_channel)
feature_stub = embeddings_pb2_grpc.EmbeddingServiceStub(feature_channel)


# Pydantic models for API responses
class MovieMetadata(BaseModel):
    movieId: int
    title: str
    releaseYear: int
    genres: List[str]
    numRatings: int


class RecommendationResponse(BaseModel):
    movies: List[MovieMetadata]


class LikeDislikeResponse(BaseModel):
    success: bool
    message: str


class LikeDislikeRequest(BaseModel):
    user_id: int
    movie_id: int


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "gateway-service"}


@app.get("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(user_id: int = Query(..., description="User ID")):
    """
    Get recommended movies for a user.
    
    Fetches ranked recommendations from the ranking service and enriches them
    with movie metadata from the feature service.
    """
    try:
        # Get ranked movie IDs from ranking service
        ranking_request = ranking_pb2.UserId(userId=user_id)
        ranking_response = ranking_stub.GetRankedRecommendations(ranking_request, timeout=10.0)
        movie_ids = list(ranking_response.movieIds)
        
        if not movie_ids:
            return RecommendationResponse(movies=[])
        
        # Fetch metadata for all recommended movies
        movies = []
        for movie_id in movie_ids:
            try:
                metadata_request = embeddings_pb2.MovieId(movieId=movie_id)
                metadata_response = feature_stub.GetMovieMetadata(metadata_request, timeout=5.0)
                
                movies.append(MovieMetadata(
                    movieId=movie_id,
                    title=metadata_response.title,
                    releaseYear=metadata_response.releaseYear,
                    genres=list(metadata_response.genres),
                    numRatings=metadata_response.numRatings
                ))
            except grpc.RpcError as e:
                # If we can't get metadata for a movie, skip it
                print(f"Warning: Failed to get metadata for movie {movie_id}: {e.details()}")
                continue
        
        return RecommendationResponse(movies=movies)
        
    except grpc.RpcError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recommendations: {e.details()}"
        )


@app.post("/like", response_model=LikeDislikeResponse)
async def like_movie(request: LikeDislikeRequest):
    """
    Like a movie for a user.
    
    This will update the user's embedding and add the movie to their watched list.
    """
    try:
        like_request = embeddings_pb2.UserIdAndMovieId(userId=request.user_id, movieId=request.movie_id)
        feature_stub.LikeMovie(like_request, timeout=5.0)
        
        return LikeDislikeResponse(
            success=True,
            message=f"Successfully liked movie {request.movie_id}"
        )
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.NOT_FOUND:
            raise HTTPException(
                status_code=404,
                detail=f"Movie {request.movie_id} not found"
            )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to like movie: {e.details()}"
        )


@app.post("/dislike", response_model=LikeDislikeResponse)
async def dislike_movie(request: LikeDislikeRequest):
    """
    Dislike a movie for a user.
    
    This will update the user's embedding and add the movie to their watched list.
    """
    try:
        dislike_request = embeddings_pb2.UserIdAndMovieId(userId=request.user_id, movieId=request.movie_id)
        feature_stub.DislikeMovie(dislike_request, timeout=5.0)
        
        return LikeDislikeResponse(
            success=True,
            message=f"Successfully disliked movie {request.movie_id}"
        )
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.NOT_FOUND:
            raise HTTPException(
                status_code=404,
                detail=f"Movie {request.movie_id} not found"
            )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to dislike movie: {e.details()}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

