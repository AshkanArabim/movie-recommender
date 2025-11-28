import os
import sys
import tempfile
from concurrent import futures

import grpc
from grpc_tools import protoc

from service import RankingService, create_servicer, _get_candidates_proto_modules, _get_embeddings_proto_modules

# Candidate service connection
CANDIDATE_SERVICE_HOST = os.environ.get('CANDIDATE_SERVICE_HOST', 'localhost')
CANDIDATE_SERVICE_PORT = os.environ.get('CANDIDATE_SERVICE_PORT', '50052')

# Feature service connection
FEATURE_SERVICE_HOST = os.environ.get('FEATURE_SERVICE_HOST', 'localhost')
FEATURE_SERVICE_PORT = os.environ.get('FEATURE_SERVICE_PORT', '60000')


def generate_proto_modules():
    """
    Generate Python modules from the proto file.
    Returns the generated modules (ranking_pb2, ranking_pb2_grpc).
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proto_file = os.path.join(script_dir, 'ranking.proto')
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
    import ranking_pb2
    import ranking_pb2_grpc
    
    return ranking_pb2, ranking_pb2_grpc


def serve():
    """Start the gRPC server."""
    # Generate proto modules
    ranking_pb2, ranking_pb2_grpc = generate_proto_modules()
    
    # Generate candidate service proto modules (needed for client stub)
    candidates_pb2, candidates_pb2_grpc = _get_candidates_proto_modules()
    
    # Generate feature service proto modules (needed for client stub)
    embeddings_pb2, embeddings_pb2_grpc = _get_embeddings_proto_modules()
    
    # Create gRPC channel to candidate service
    candidate_service_address = f'{CANDIDATE_SERVICE_HOST}:{CANDIDATE_SERVICE_PORT}'
    print(f"Connecting to candidate service at {candidate_service_address}...")
    candidate_channel = grpc.insecure_channel(candidate_service_address)
    
    # Wait for channel to be ready (with timeout)
    try:
        grpc.channel_ready_future(candidate_channel).result(timeout=10)
        print("Connected to candidate service.")
    except grpc.FutureTimeoutError:
        print(f"Warning: Could not connect to candidate service at {candidate_service_address}")
        print("The service will attempt to connect when needed.")
    
    # Create candidate service stub
    candidate_service_stub = candidates_pb2_grpc.CandidateServiceStub(candidate_channel)
    
    # Create gRPC channel to feature service
    feature_service_address = f'{FEATURE_SERVICE_HOST}:{FEATURE_SERVICE_PORT}'
    print(f"Connecting to feature service at {feature_service_address}...")
    feature_channel = grpc.insecure_channel(feature_service_address)
    
    # Wait for channel to be ready (with timeout)
    try:
        grpc.channel_ready_future(feature_channel).result(timeout=10)
        print("Connected to feature service.")
    except grpc.FutureTimeoutError:
        print(f"Warning: Could not connect to feature service at {feature_service_address}")
        print("The service will attempt to connect when needed.")
    
    # Create feature service stub
    feature_service_stub = embeddings_pb2_grpc.EmbeddingServiceStub(feature_channel)
    
    # Create ranking service instance
    ranking_service = RankingService(candidate_service_stub, feature_service_stub, ranking_pb2)
    servicer = create_servicer(ranking_service, ranking_pb2_grpc)
    
    # Start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    ranking_pb2_grpc.add_RankingServiceServicer_to_server(servicer, server)
    grpc_port = os.environ.get('GRPC_PORT', '50053')
    server.add_insecure_port(f'[::]:{grpc_port}')
    print(f"gRPC RankingService serving on port {grpc_port}")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()

