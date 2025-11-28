# Client Service

Simple frontend client for the movie recommendation system.

## Features

- Landing page with movie recommendations
- Movie detail page with like/dislike functionality
- Automatic refresh of recommendations when returning to the main page

## Configuration

The user ID is hardcoded in `app.js` at the top of the file:

```javascript
const USER_ID = 0;
```

To change the user ID, simply modify this constant.

The gateway URL is also configurable:

```javascript
const GATEWAY_URL = 'http://localhost:8000';
```

## Running the Client

### Option 1: Using Docker Compose

The client service is included in `docker-compose.yaml`. Start all services:

```bash
docker-compose up
```

Then access the client at: http://localhost:8080

### Option 2: Local Development

You can serve the client locally using Python's HTTP server:

```bash
cd client-service
python -m http.server 8080
```

Then open http://localhost:8080 in your browser.

### Option 3: Direct File Access

You can also open `index.html` directly in your browser, but note that CORS may prevent API calls if the gateway is running on a different origin.

## Usage

1. **View Recommendations**: The landing page automatically loads and displays recommended movies when opened.

2. **View Movie Details**: Click on any movie card to see details and like/dislike options.

3. **Like/Dislike**: Click the "Like" or "Dislike" button on a movie's detail page. After the action completes, you'll be automatically returned to the recommendations page, which will refresh with new recommendations.

