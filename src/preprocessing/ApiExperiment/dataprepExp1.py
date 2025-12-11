import pandas as pd
import sys
import os
import pandas as pd
from tmdbv3api import TMDb, Movie, TV
import re
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpersfunc import handle_duplicate_movies, encode_genres, concat_ratings_movies,drop_missing_genres,check_and_clean_consistency

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))

movies_df = pd.read_csv(r'c:\Users\maata\Desktop\E-commerce\data\raw\movies.csv')
ratings_df = pd.read_csv(r'c:\Users\maata\Desktop\E-commerce\data\raw\ratings.csv')

# SETUP
tmdb = TMDb()
tmdb.api_key = os.getenv('TMDB_API_KEY')
movie_api = Movie()
tv_api = TV()

df_api = movies_df.copy()

def get_real_genre(row):
    if row['genres'] == '(no genres listed)':
        original_title = row['title']

        # --- AGGRESSIVE CLEANING ---
        clean_title = re.sub(r'\s\(.*\)$', '', original_title)
        if ':' in clean_title:
            clean_title = clean_title.split(':')[0]
        clean_title = clean_title.strip()
        # ---------------------------

        # PHASE 1: Try Movie Search
        try:
            search_movie = movie_api.search(clean_title)
            if search_movie:
                details = movie_api.details(search_movie[0].id)
                genres = [g['name'] for g in details.genres]
                print(f"Movie found: '{clean_title}' -> {genres}")
                return "|".join(genres)
        except Exception:
            # If Movie search crashes, just ignore it and move to TV
            pass

        # PHASE 2: Try TV Search (If Movie failed or crashed)
        try:
            search_tv = tv_api.search(clean_title)
            if search_tv:
                details = tv_api.details(search_tv[0].id)
                genres = [g['name'] for g in details.genres]
                print(f"TV Show found: '{clean_title}' -> {genres}")
                return "|".join(genres)
        except Exception:
            pass

        print(f"Not Found: '{clean_title}'")
        return row['genres']

    return row['genres']

# Apply
def API_genre_exp():
    global ratings_df
    print("Starting Final Unstoppable Fetching...")
    mask = df_api['genres'] == '(no genres listed)'
    df_api.loc[mask, 'genres'] = df_api.loc[mask].apply(get_real_genre, axis=1)
    print("Fetching Complete.")
    print(df_api[df_api['genres'] == '(no genres listed)'])
    # Check the final 2
    print("\n--- Remaining Missing ---")
    print(df_api[df_api['genres'] == '(no genres listed)'])

    df_api_cleaned = handle_duplicate_movies(df_api)
    df_api_cleaned = drop_missing_genres(df_api_cleaned)
    ratings_df = check_and_clean_consistency(ratings_df,df_api_cleaned, clean=True)
    df_api_encoded = encode_genres(df_api_cleaned, column_name='genres')
    print("\n--- Data After Cleaning and Encoding ---")
    print(df_api_encoded.head(10))
    df_final = concat_ratings_movies(df_api_encoded, ratings_df)
    CLEAN_DATASET_EXP1_CSV = r'c:\Users\maata\Desktop\E-commerce\data\processed\clean_dataset_expAPI.csv'
    df_final.to_csv(CLEAN_DATASET_EXP1_CSV, index=False)
    return df_final