#!/usr/bin/python3
"""
Movie Bot - Recommend movies based on user preferences

The job of movie bot is to read the history of previous movies recommend and then suggestion a new
movie based on the user's preferences. The bot will also store the new movie recommendation in the
database for future reference.

    * The bot will identify the genre of the movie it recommends from a set of predefined genres from
      the database.
    * The bot will use the LLM to generate the movie recommendation.
    * The bot will store the new movie recommendation in the database.
    * The bot will return the movie recommendation to the user along with the genre.
    # The bot will attempt to recommend a movie that is not already in the database and
      will try to vary the genre of the movie recommendation.

Author: Jason A. Cox
2 Feb 2025
https://github.com/jasonacox/TinyLLM

"""

import openai
import sqlite3
import os
import time
import datetime
import random

# Version
VERSION = "v0.0.1"

# Configuration Settings
api_key = os.environ.get("OPENAI_API_KEY", "open_api_key")                  # Required, use bogus string for Llama.cpp
api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")    # Required, use https://api.openai.com for OpenAI
mymodel = os.environ.get("LLM_MODEL", "models/7B/gguf-model.bin")           # Pick model to use e.g. gpt-3.5-turbo for OpenAI
DATABASE = os.environ.get("DATABASE", "moviebot.db")                        # Database file to store movie data
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"                  # Set to True to enable debug mode
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.7))                     # LLM temperature
MESSAGE_FILE = os.environ.get("MESSAGE_FILE", "message.txt")                # File to store the message
ABOUT_ME = os.environ.get("ABOUT_ME",
         "We love movies! Action, adventure, sci-fi and feel good movies are our favorites.")  # About me text

#DEBUG = True

# Debugging functions
def log(text):
    # Print to console
    if DEBUG:
        print(f"INFO: {text}")

def error(text):
    # Print to console
    print(f"ERROR: {text}")

class MovieDatabase:
    def __init__(self, db_name=DATABASE):
        self.db_name = db_name
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_name)
        return self.conn

    def create(self):
        self.connect()
        c = self.conn.cursor()
        # Create table to store movies
        c.execute('''CREATE TABLE IF NOT EXISTS movies (title text, genre text, date_recommended text)''')
        self.conn.commit()
        # Create table to store genres (make them unique)
        c.execute('''CREATE TABLE IF NOT EXISTS genres (genre text UNIQUE)''')
        self.conn.commit()
        self.conn.close()

    def destroy(self):
        self.connect()
        c = self.conn.cursor()
        # Drop tables
        c.execute('''DROP TABLE IF EXISTS movies''')
        self.conn.commit()
        c.execute('''DROP TABLE IF EXISTS genres''')
        self.conn.commit()
        self.conn.close()

    def insert_movie(self, title, genre, date_recommended=None):
        if date_recommended is None:
            date_recommended = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.connect()
        c = self.conn.cursor()
        c.execute("INSERT INTO movies VALUES (?, ?, ?)", (title, genre, date_recommended))
        self.conn.commit()
        self.conn.close()

    def select_all_movies(self, sort_by='date_recommended'):
        self.connect()
        c = self.conn.cursor()
        c.execute(f"SELECT * FROM movies ORDER BY {sort_by}")
        rows = c.fetchall()
        self.conn.close()
        # return list of movie titles
        m = [row[0] for row in rows]
        return m

    def select_movies_by_genre(self, genre):
        self.connect()
        c = self.conn.cursor()
        c.execute(f"SELECT * FROM movies WHERE genre = ?", (genre,))
        rows = c.fetchall()
        self.conn.close()
        # return list of movie titles
        m = [row[0] for row in rows]
        return m

    def delete_all_movies(self):
        self.connect()
        c = self.conn.cursor()
        c.execute("DELETE FROM movies")
        self.conn.commit()
        self.conn.close()

    def delete_movie_by_title(self, title):
        self.connect()
        c = self.conn.cursor()
        c.execute("DELETE FROM movies WHERE title = ?", (title,))
        self.conn.commit()
        self.conn.close()

    def insert_genre(self, genre):
        self.connect()
        c = self.conn.cursor()
        c.execute("INSERT OR IGNORE INTO genres VALUES (?)", (genre,))
        self.conn.commit()
        self.conn.close()

    def select_genres(self):
        self.connect()
        c = self.conn.cursor()
        c.execute("SELECT * FROM genres")
        rows = c.fetchall()
        self.conn.close()
        # convert payload into a list of genres
        g = [row[0] for row in rows]
        return g

# Initialize the database
db = MovieDatabase()
#db.destroy()  # Clear the database
db.create()  # Set up the database if it doesn't exist

# Load genres
genres = db.select_genres()
if len(genres) == 0:
    for genre in ['action', 'fantasy', 'comedy', 'drama', 'horror', 'romance', 
                  'sci-fi', 'christmas', 'musical', 'thriller', 'mystery',
                  'animated', 'western', 'documentary', 'adventure', 'family']:
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

# LLM Function to ask the chatbot a question
def ask_chatbot(prompt):
    log(f"LLM: Asking chatbot: {prompt}")
    response = llm.chat.completions.create(
        model=mymodel,
        stream=False,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    log(f"LLM: Response: {response.choices[0].message.content.strip()}\n")
    return response.choices[0].message.content.strip()

# Function to recommend a movie
def recommend_movie():
    # Get list of movies but limit it to last 50 
    movies = db.select_all_movies(sort_by='date_recommended')[-50:]
    # Get list of genres
    genres = db.select_genres()
    # Randomly mix up the elements in the list
    random.shuffle(genres)

    # Get today's date in format: Month Day, Year
    today = datetime.datetime.now().strftime("%B %d, %Y")

    prompt = f"""You are a expert movie bot that recommends movies based on user preferences.
        Today is {today}, in case it is helpful in suggesting a movie.
        Provide variety and interest.
        You have recommended the following movies in the past:

        {movies}

        You have the following genres available:
        {genres}

        {ABOUT_ME}
        Please recommend a new movie to watch. List only one. Try not to repeat something recommended already. 
        Give interesting details about the movie.
        """

    # Ask the chatbot for a movie recommendation
    recommendation = ask_chatbot(prompt).strip()

    # Save the movie recommendation to file
    with open(MESSAGE_FILE, "w") as f:
        f.write(recommendation)

    # Ask the chatbot to provide just one move and the title only
    prompt = f"""About me: {ABOUT_ME}
    
    Recommendation:
    {recommendation}

    Based on the above recommendation, select the best movie for me. List the movie title only:
    """
    movie_recommendation = ask_chatbot(prompt).strip()

    # Ask the chatbot what type of genre the movie is, provide list of genres
    prompt = f"""You are a expert movie bot that recommends movies based on user preferences.
        You have the following genres available:
        {genres}
        
        Based on that list, what genre is the movie "{movie_recommendation}"? List the genre only:
        """
    genre = ask_chatbot(prompt).strip().lower()

    # Store the movie recommendation
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

    if len(os.sys.argv) > 1:
        print(f"Movie Bot {VERSION} - Command Line Interface - DB: {DATABASE}\n")
        cmd = os.sys.argv[1].lower()
        if cmd in commands:
            commands[cmd]()
        else:
            commands["help"]()
        print("")
        os.sys.exit(1)

    # Default action
    movie, genre = recommend_movie()
    output = f"Movie Bot recommends the {genre} movie: {movie}"
    print(output)
