# Movie Recommender

A microservices-based movie recommendation system using collaborative filtering with embeddings, built on gRPC and FastAPI.

## Architecture

<img width="885" height="755" alt="image" src="https://github.com/user-attachments/assets/92dd4c5f-1ca1-4487-b6d4-59aa5b5bccf7" />

- **Client:** Vanilla JS frontend that lets users browse recommendations and like/dislike movies.
- **Gateway:** FastAPI REST service that exposes `GET /recommendations`, `POST /like`, and `POST /dislike`. Communicates with backend services over gRPC.
- **Feature Service:** Source of truth for 64-dimensional user/movie embeddings. Updates user embeddings in real-time based on feedback using an exponential moving average. Backed by PostgreSQL + Redis.
- **Candidate Service:** Runs FAISS similarity search against movie embeddings to return the top 50 candidates for a given user. GPU-accelerated when available.
- **Ranking Service:** Scores candidates by a weighted combination of popularity and recency, returning the top 15.

## Setup

### Prerequisites

- Docker and Docker Compose
- MovieLens 25M dataset — download from [grouplens.org](https://grouplens.org/datasets/movielens/25m/) and extract into `./data/`
- Pre-computed embeddings (`movie_embeddings.pkl`, `user_embeddings.pkl`) placed in `./data/` — generate using `embedding-calc/exp.ipynb` if not present

### Running

```bash
docker-compose up --build
```

- Web interface: http://localhost:8080
- Gateway API + docs: http://localhost:8000/docs

```bash
docker-compose down      # stop
docker-compose down -v   # stop and clear database
```
