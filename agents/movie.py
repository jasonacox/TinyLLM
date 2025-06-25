#!/usr/bin/python3
"""
Movie Bot - Recommend movies based on user preferences and current events

Movie Bot reads the history of previous movie recommendations and suggests a new
movie based on the user's preferences, current date, and internet search results.
The bot stores new movie recommendations in a local database for future
reference.

Features:
    * Identifies the genre of the recommended movie from a set of predefined 
      genres in the database.
    * Uses an LLM (Large Language Model) to generate movie recommendations. 
    * Integrates with a local SearxNG instance to search the internet for 
      relevant and recent movies, including those related to current holidays, 
      seasons, or events.
    * Extracts movie titles from internet search results using the LLM for 
      improved accuracy.
    * Avoids recommending movies already in the database and tries to vary the 
      genre.
    * Stores new movie recommendations and their genres in the database.
    * Returns the movie recommendation and genre to the user.

Author: Jason A. Cox
2 Feb 2025
https://github.com/jasonacox/TinyLLM

"""

import openai
import sqlite3
import os
import sys
import time
import datetime
import random
import requests

# Version
VERSION = "v0.0.2"

# Configuration Settings
api_key = os.environ.get("OPENAI_API_KEY", "open_api_key")                  # Required, use bogus string for Llama.cpp
api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")    # Required, use https://api.openai.com for OpenAI
mymodel = os.environ.get("LLM_MODEL", "models/7B/gguf-model.bin")           # Pick model to use e.g. gpt-3.5-turbo for OpenAI
DATABASE = os.environ.get("DATABASE", "moviebot.db")                        # Database file to store movie data
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"                  # Set to True to enable debug mode
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.7))                     # LLM temperature
MESSAGE_FILE = os.environ.get("MESSAGE_FILE", "message.txt")                # File to store the message
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:6060/")       # SearxNG instance URL
ABOUT_ME = os.environ.get("ABOUT_ME",
         "We love movies! Action, adventure, sci-fi and feel good movies are our favorites.")  # About me text

# Debugging functions
def log(text):
    """Log informational messages if DEBUG is enabled."""
    if DEBUG:
        print(f"INFO: {text}")
        sys.stdout.flush()

def error(text):
    """Print error messages to the console."""
    print(f"ERROR: {text}")

class MovieDatabase:
    def __init__(self, db_name=DATABASE):
        """Initialize the MovieDatabase with the given database name."""
        self.db_name = db_name
        self.conn = None

    def connect(self):
        """Connect to the SQLite database and return the connection object."""
        self.conn = sqlite3.connect(self.db_name)
        return self.conn

    def create(self):
        """Create the movies and genres tables if they do not exist."""
        self.connect()
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS movies (title text, genre text, date_recommended text)''')
        self.conn.commit()
        c.execute('''CREATE TABLE IF NOT EXISTS genres (genre text UNIQUE)''')
        self.conn.commit()
        self.conn.close()

    def destroy(self):
        """Drop the movies and genres tables from the database."""
        self.connect()
        c = self.conn.cursor()
        c.execute('''DROP TABLE IF EXISTS movies''')
        self.conn.commit()
        c.execute('''DROP TABLE IF EXISTS genres''')
        self.conn.commit()
        self.conn.close()

    def insert_movie(self, title, genre, date_recommended=None):
        """Insert a new movie with its genre and recommendation date into the database."""
        if date_recommended is None:
            date_recommended = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.connect()
        c = self.conn.cursor()
        c.execute("INSERT INTO movies VALUES (?, ?, ?)", (title, genre, date_recommended))
        self.conn.commit()
        self.conn.close()

    def select_all_movies(self, sort_by='date_recommended'):
        """Return a list of all movie titles, sorted by the specified column."""
        self.connect()
        c = self.conn.cursor()
        c.execute(f"SELECT * FROM movies ORDER BY {sort_by}")
        rows = c.fetchall()
        self.conn.close()
        # return list of movie titles
        m = [row[0] for row in rows]
        return m

    def select_movies_by_genre(self, genre):
        """Return a list of movie titles for the specified genre."""
        self.connect()
        c = self.conn.cursor()
        c.execute(f"SELECT * FROM movies WHERE genre = ?", (genre,))
        rows = c.fetchall()
        self.conn.close()
        # return list of movie titles
        m = [row[0] for row in rows]
        return m

    def delete_all_movies(self):
        """Delete all movies from the database."""
        self.connect()
        c = self.conn.cursor()
        c.execute("DELETE FROM movies")
        self.conn.commit()
        self.conn.close()

    def delete_movie_by_title(self, title):
        """Delete a movie from the database by its title."""
        self.connect()
        c = self.conn.cursor()
        c.execute("DELETE FROM movies WHERE title = ?", (title,))
        self.conn.commit()
        self.conn.close()

    def insert_genre(self, genre):
        """Insert a new genre into the database if it does not already exist."""
        self.connect()
        c = self.conn.cursor()
        c.execute("INSERT OR IGNORE INTO genres VALUES (?)", (genre,))
        self.conn.commit()
        self.conn.close()

    def select_genres(self):
        """Return a list of all genres in the database."""
        self.connect()
        c = self.conn.cursor()
        c.execute("SELECT * FROM genres")
        rows = c.fetchall()
        self.conn.close()
        # convert payload into a list of genres
        g = [row[0] for row in rows]
        return g

# --- Constants ---
GENRE_LIST = [
    'action', 'fantasy', 'comedy', 'drama', 'horror', 'romance',
    'sci-fi', 'christmas', 'musical', 'thriller', 'mystery',
    'animated', 'western', 'documentary', 'adventure', 'family'
]

# Initialize the database
db = MovieDatabase()
#db.destroy()  # Clear the database
db.create()  # Set up the database if it doesn't exist

# Load genres
genres = db.select_genres()
if len(genres) == 0:
    for genre in GENRE_LIST:
        db.insert_genre(genre)
    genres = db.select_genres()

# Load movies
movies = db.select_all_movies()
if len(movies) == 0:
    # Load some sample movies
    db.insert_movie("Home Alone", "christmas")
    db.insert_movie("Sound of Music", "musical")
    db.insert_movie("Die Hard", "action")
    db.insert_movie("Sleepless in Seattle", "romance")
    db.insert_movie("The Game", "thriller")
    db.insert_movie("Toy Story", "animated")
    db.insert_movie("Sherlock Holmes", "mystery")
    db.insert_movie("Airplane", "comedy")
    db.insert_movie("Raiders of the Lost Ark", "action")
    db.insert_movie("A Bug's Life", "drama")

# Test LLM
while True:
    log("Testing LLM...")
    try:
        log(f"Using openai library version {openai.__version__}")
        log(f"Connecting to OpenAI API at {api_base} using model {mymodel}")
        llm = openai.OpenAI(api_key=api_key, base_url=api_base)
        # Get models
        models = llm.models.list()
        # build list of models
        model_list = [model.id for model in models.data]
        log(f"LLM: Models available: {model_list}")
        if len(models.data) == 0:
            log("LLM: No models available - check your API key and endpoint.")
            raise Exception("No models available")
        if not mymodel in model_list:
            log(f"LLM: Model {mymodel} not found in models list.")
            if len(model_list) == 1:
                log("LLM: Switching to default model")
                mymodel = model_list[0]
            else:
                log(f"LLM: Unable to find model {mymodel} in models list.")
                raise Exception(f"Model {mymodel} not found")
        log(f"LLM: Using model {mymodel}")
        # Test LLM
        llm.chat.completions.create(
            model=mymodel,
            stream=False,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": "Hello"}],
        )
        break
    except Exception as e:
        log("OpenAI API Error: %s" % e)
        log(f"Unable to connect to OpenAI API at {api_base} using model {mymodel}.")
        log("Sleeping 10 seconds...")
        time.sleep(10)

# Pull movies released today from MoviesThisDay.com API
# # Movies released today (JSON)
# curl -X GET 'http://moviesthisday.com/movies/today'
# {
#     "movies": [
#         {
#             "id": "8587",
#             "title": "The Lion King",
#             "release_date": "1994-06-24",
#             "original_language": "en",
#             "popularity": "87.384",
#             "vote_average": "8.256",
#             "vote_count": "16991",
#             "imdb_id": "tt0110357",
#             "language": "en",
#             "release_year": "1994",
#             "runtime": 89,
#             "omdb_genre": "Animation, Adventure, Drama",
#             "omdb_imdb_rating": "8.5",
#             "omdb_imdb_votes": "1,204,496",
#             "omdb_box_office": "$424,979,720",
#             "production_companies": "Walt Disney Pictures, Walt Disney Feature Animation",
#             "omdb_poster": "https://m.media-amazon.com/x.jpg",
#             "omdb_plot": "Lion prince Simba and his father are targeted by his bitter uncle, who wants to ascend the throne himself.",
#             "omdb_rated": "G",
#             "popularity_rank": 249,
#             "is_new_release": false
#         }
#     ]
# }...
def get_movies_released_today(movie_only=False):
    """
    Fetch movies released today from the local MoviesThisDay API.

    Args:
        movie_only (bool):
            - If True, return a list of movie names (titles) only (popularity > 10).
            - If False, return a string prompt suitable for RAG context, listing title, year, rated, genre, and popularity.

    Returns:
        list[str]: If movie_only is True, a list of movie titles with popularity > 10.
        str: If movie_only is False, a formatted string for RAG context.
        [] or str: If no movies found or error, returns [] (if movie_only) or error string (if not).
    """
    url = "http://moviesthisday.com/movies/today"
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        movies = data.get("movies", [])
        if not movies:
            # No movies found for today
            return [] if movie_only else None
        # Filter movies by popularity > 10
        filtered_movies = [m for m in movies if float(m.get('popularity', 0)) >= 10]
        if movie_only:
            # Return only the movie titles
            return [m.get('title', 'Unknown') for m in filtered_movies]
        # Build a formatted string for RAG context
        movie_lines = []
        for m in filtered_movies:
            title = m.get('title', 'Unknown')
            year = m.get('release_year', '')
            rated = m.get('omdb_rated', 'NR')
            genre = m.get('omdb_genre', 'Unknown')
            popularity = float(m.get('popularity', 0))
            # Format each movie's info as a line
            line = f"- {title} ({year}), Rated: {rated}, Genre: {genre}, Popularity: {popularity:.1f}"
            movie_lines.append(line)
        if not movie_lines:
            return f"No popular movies (popularity > 10) found released on {today}."
        context = (
            f"Today's date is {today}. Here are movies released on this day in history with popularity over 10, which you can use to recommend a movie:\n\n"
            + "\n".join(movie_lines)
        )
        return context
    except Exception as e:
        # Handle network or parsing errors
        return [] if movie_only else f"Could not retrieve today's movie releases due to error: {e}"

# LLM Function to ask the chatbot a question
def ask_chatbot(prompt):
    """Send a prompt to the LLM chatbot and return its response as a string."""
    log(f"LLM: Asking chatbot: {prompt}")
    response = llm.chat.completions.create(
        model=mymodel,
        stream=False,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    log(f"LLM: Response: {response.choices[0].message.content.strip()}\n")
    return response.choices[0].message.content.strip()

def get_searxng_movies(query, max_results=10):
    """Query local SearxNG for relevant movie results and return a simple text list of 'title - content' lines."""
    searxng_url = SEARXNG_URL
    params = {
        "q": query,
        "categories": "videos,news",
        "format": "json",
        "language": "en",
        "safesearch": 1,
    }
    try:
        resp = requests.get(f"{searxng_url}/search", params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json()
        lines = []
        for r in results.get("results", [])[:max_results]:
            title = r.get("title", "").strip()
            content = r.get("content", r.get("snippet", "")).strip()
            if title or content:
                lines.append(f"{title} - {content}")
        return "\n".join(lines)
    except Exception as e:
        log(f"SearxNG search failed: {e}")
        return ""

def recommend_movie():
    """Recommend a new movie using the LLM, considering user history, genres, 
    and SearxNG search results."""
    # Get list of movies but limit it to last 50
    log("Selecting last 50 movies from the database...")
    movies = db.select_all_movies(sort_by='date_recommended')[-50:]
    genres = db.select_genres()
    random.shuffle(genres)
    today = datetime.datetime.now().strftime("%B %d, %Y")
    # Holiday/season detection using a list of tuples instead of a dict with lists as keys
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day
    holiday = None
    holiday_ranges = [
        (12, range(20, 32), "Christmas"),
        (10, range(25, 32), "Halloween"),
        (7, [4], "Independence Day"),
        (2, [14], "Valentine's Day"),
        (3, range(15, 21), "Spring"),
        (11, range(20, 31), "Thanksgiving")
    ]
    for m, days, name in holiday_ranges:
        if month == m and day in days:
            holiday = name
            break
    if not holiday:
        if 6 <= month <= 8:
            holiday = "Summer"
        elif 9 <= month <= 11:
            holiday = "Fall"
        elif month <= 3 or month >= 10:
            holiday = "Winter"
    # Build search terms based on current date and holiday
    search_terms = ["great movies to watch today",
                    "new movies available for streaming", 
                    "popular movies right now",
                    "streaming movies great to watch before going to theaters"]
    if holiday:
        search_terms.append(f"best {holiday} movies")
    searxng_results = ""
    log(f"Searching for movies using SearxNG with terms: {search_terms}")
    for term in search_terms:
        search_results = get_searxng_movies(term)
        if not search_results:
            break  # Stop if no results found
        searxng_results += f"{term}:\n" + search_results + "\n\n"
    # format: June 13th
    log(f"Search results from SearxNG:\n{searxng_results}")
    search_date = datetime.datetime.now().strftime("%B %d")
    #prompt = f"List some movies released on {search_date} in history."
    #movies_this_day = ask_chatbot(prompt)
    #log(f"Movies released on {search_date}: {movies_this_day}")
    #searxng_results += f"{prompt}:\n" + movies_this_day + "\n\n"
    # Let the LLM extract movie titles from the raw search results
    prompt_extract = f"""Here are some internet search results about movies:\n\n
    {searxng_results}\n\n
    List the movie titles you see in the above list, one per line. 
    Give a brief description of each movie.
    Do not include any movies that are already in the database: {movies}

    Do not include titles that are about lists and not actual movies.
    """
    log(f"Extracting movie titles from search results: {prompt_extract}")
    # Ask the chatbot to extract movie titles from the search results
    extracted_movies = ask_chatbot(prompt_extract)
    log(f"Extracted movie titles:\n{extracted_movies}")
    # Get list of movies released on this date in history
    movies_this_day = get_movies_released_today() or ""
    if movies_this_day:
        log(f"Movies released on {search_date} in history:\n{movies_this_day}")
        movies_this_day = f"\nHere are some movies released on this date in history, {search_date}:\n" + movies_this_day
    # Compose main prompt with extracted movie titles
    prompt = f"""You are an expert movie bot that recommends movies based on user 
    preferences and current events.

    Today is {today} ({holiday} time).
    Do not recommend any of these movies: {movies}
    {movies_this_day}
    
    Internet search found these movie titles: 
    {extracted_movies}

    {ABOUT_ME}
    
    Background:
    List new movies and popular movies.
    List movies released on this date in history.

    Recommendation:
    Please recommend a new movie to watch. List only one.
    Give interesting details about the movie.
    """
    recommendation = ask_chatbot(prompt).strip()
    log(f"Movie recommendation: {recommendation}")
    log("------------------------------------\n\n")
    # Save the recommendation to a file
    with open(MESSAGE_FILE, "w") as f:
        f.write(recommendation)
    prompt = f"""Recommendation:\n{recommendation}\n\nBased on the above, what is the recommended movie? List the movie title only:\n"""
    movie_recommendation = ask_chatbot(prompt).strip()
    log(f"Movie recommendation title: {movie_recommendation}")
    prompt = f"""You are an expert movie bot that recommends movies based on user preferences.\nYou have the following genres available: {genres}\nBased on that list, what genre is the movie \"{movie_recommendation}\"? List the genre only:\n"""
    genre = ask_chatbot(prompt).strip().lower()
    log(f"Movie genre: {genre}")
    db.insert_movie(movie_recommendation, genre)
    return movie_recommendation, genre

# Main
if __name__ == "__main__":
    log(f"Movie Bot {VERSION} - Recommends movies based on user preferences and history.")
    
    # Command line argument handling
    commands = {
        "clear": lambda: (db.delete_all_movies(), print("Database cleared.")),
        "history": lambda: (
            db.connect(),
            print("\n".join([f"{row[2]} - {row[0]}" for row in 
                  db.conn.cursor().execute("SELECT * FROM movies ORDER BY date_recommended").fetchall()])),
            db.conn.close()
        ),
        "genres": lambda: print("\n".join(db.select_genres())),
        "help": lambda: print("Usage: python movie.py [clear|history|genres|help]")
    }

    if len(sys.argv) > 1:
        print(f"Movie Bot {VERSION} - Command Line Interface - DB: {DATABASE}\n")
        cmd = sys.argv[1].lower()
        if cmd in commands:
            commands[cmd]()
        else:
            commands["help"]()
        print("")
        sys.exit(1)

    # Default action
    suffix = ""
    movie, genre = recommend_movie()
    if movie in get_movies_released_today(movie_only=True):
        suffix = " (released this day)"
    output = f"Movie Bot recommends the {genre} movie: {movie}{suffix}"
    log("\n\n\n------------------------------------")
    print(output)
