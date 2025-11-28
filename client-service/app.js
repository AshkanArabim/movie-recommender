// Configuration - Easy to change user ID later
const USER_ID = 0;
const GATEWAY_URL = 'http://localhost:8000';

// State
let currentMovieId = null;
let recommendations = [];

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('user-id').textContent = USER_ID;
    loadRecommendations();
});

// API Functions
async function fetchRecommendations() {
    try {
        const response = await fetch(`${GATEWAY_URL}/recommendations?user_id=${USER_ID}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return data.movies || [];
    } catch (error) {
        console.error('Error fetching recommendations:', error);
        throw error;
    }
}

async function likeMovie() {
    if (!currentMovieId) return;
    
    try {
        const response = await fetch(`${GATEWAY_URL}/like`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: USER_ID,
                movie_id: currentMovieId
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        showFeedback('Movie liked! Returning to recommendations...', true);
        
        // Return to landing page after a short delay
        setTimeout(() => {
            goToLanding();
        }, 1500);
    } catch (error) {
        console.error('Error liking movie:', error);
        showFeedback('Failed to like movie. Please try again.', false);
    }
}

async function dislikeMovie() {
    if (!currentMovieId) return;
    
    try {
        const response = await fetch(`${GATEWAY_URL}/dislike`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: USER_ID,
                movie_id: currentMovieId
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        showFeedback('Movie disliked! Returning to recommendations...', true);
        
        // Return to landing page after a short delay
        setTimeout(() => {
            goToLanding();
        }, 1500);
    } catch (error) {
        console.error('Error disliking movie:', error);
        showFeedback('Failed to dislike movie. Please try again.', false);
    }
}

// UI Functions
async function loadRecommendations() {
    const loadingEl = document.getElementById('loading');
    const errorEl = document.getElementById('error');
    const containerEl = document.getElementById('recommendations-container');
    const recommendationsListEl = document.getElementById('recommendations-list');
    
    // Show loading, hide others
    loadingEl.classList.remove('hidden');
    errorEl.classList.add('hidden');
    containerEl.classList.add('hidden');
    
    try {
        recommendations = await fetchRecommendations();
        
        if (recommendations.length === 0) {
            recommendationsListEl.innerHTML = '<p class="no-results">No recommendations available.</p>';
        } else {
            recommendationsListEl.innerHTML = recommendations.map(movie => `
                <div class="movie-card" onclick="viewMovie(${movie.movieId})">
                    <div class="movie-card-content">
                        <h3>${escapeHtml(movie.title)}</h3>
                        <p class="movie-card-meta">
                            ${movie.releaseYear > 0 ? movie.releaseYear : 'Unknown Year'} • 
                            ${movie.genres.length > 0 ? movie.genres.join(', ') : 'No genres'} • 
                            ${movie.numRatings} ratings
                        </p>
                    </div>
                </div>
            `).join('');
        }
        
        loadingEl.classList.add('hidden');
        containerEl.classList.remove('hidden');
    } catch (error) {
        loadingEl.classList.add('hidden');
        errorEl.classList.remove('hidden');
        document.getElementById('error-message').textContent = 
            `Failed to load recommendations: ${error.message}`;
    }
}

function viewMovie(movieId) {
    const movie = recommendations.find(m => m.movieId === movieId);
    if (!movie) {
        console.error('Movie not found:', movieId);
        return;
    }
    
    currentMovieId = movieId;
    
    // Update movie detail page
    document.getElementById('detail-title').textContent = movie.title;
    document.getElementById('detail-year').textContent = 
        movie.releaseYear > 0 ? `Released: ${movie.releaseYear}` : 'Release year unknown';
    document.getElementById('detail-genres').textContent = 
        movie.genres.length > 0 ? `Genres: ${movie.genres.join(', ')}` : 'No genres listed';
    document.getElementById('detail-ratings').textContent = 
        `${movie.numRatings} ratings`;
    
    // Hide feedback
    document.getElementById('action-feedback').classList.add('hidden');
    
    // Navigate to movie page
    showPage('movie-page');
}

function goToLanding() {
    showPage('landing-page');
    // Reload recommendations when returning to landing page
    loadRecommendations();
}

function showPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.add('hidden');
        page.classList.remove('active');
    });
    
    // Show target page
    const targetPage = document.getElementById(pageId);
    targetPage.classList.remove('hidden');
    targetPage.classList.add('active');
}

function showFeedback(message, isSuccess) {
    const feedbackEl = document.getElementById('action-feedback');
    const messageEl = document.getElementById('feedback-message');
    
    messageEl.textContent = message;
    feedbackEl.classList.remove('hidden');
    feedbackEl.className = `action-feedback ${isSuccess ? 'success' : 'error'}`;
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

