import os
import sys
import tempfile
from concurrent import futures

import grpc
from grpc_tools import protoc

import redis_loader
from service import EmbeddingService


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


def serve():
    """Start the gRPC server."""
    # Generate proto modules
    embeddings_pb2, embeddings_pb2_grpc = generate_proto_modules()
    
    # Upload embeddings to Redis at startup
    print("Uploading embeddings to Redis...")
    redis_client = redis_loader.get_redis_client()
    redis_loader.upload_embeddings(redis_client)
    print("Embeddings uploaded successfully.")
    
    # Create service instance
    embedding_service = EmbeddingService(redis_client, embeddings_pb2)
    from service import create_servicer
    servicer = create_servicer(embedding_service, embeddings_pb2_grpc)
    
    # Start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    embeddings_pb2_grpc.add_EmbeddingServiceServicer_to_server(servicer, server)
    grpc_port = os.environ.get('GRPC_PORT', '50051')
    server.add_insecure_port(f'[::]:{grpc_port}')
    print(f"gRPC EmbeddingService serving on port {grpc_port}")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()

