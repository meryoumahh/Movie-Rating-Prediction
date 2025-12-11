import pandas as pd
import numpy as np

# Load the three processed datasets
df_nltk = pd.read_csv(r'c:\Users\maata\Desktop\E-commerce\data\processed\clean_dataset_expNLTK.csv')
df_api = pd.read_csv(r'c:\Users\maata\Desktop\E-commerce\data\processed\clean_dataset_expAPI.csv')
df_delete = pd.read_csv(r'c:\Users\maata\Desktop\E-commerce\data\processed\clean_dataset_expDELETE.csv')

print("=== DATASET COMPARISON ===\n")

# Basic info about each dataset
print("Dataset Shapes:")
print(f"NLTK: {df_nltk.shape}")
print(f"API: {df_api.shape}")
print(f"DELETE: {df_delete.shape}\n")

# Top 5 rated movies from each dataset
print("=== TOP 5 RATED MOVIES ===\n")

print("NLTK Dataset:")
top5_nltk = df_nltk.nlargest(5, 'avg_rating')[['title', 'avg_rating', 'genres']]
print(top5_nltk)
print()

print("API Dataset:")
top5_api = df_api.nlargest(5, 'avg_rating')[['title', 'avg_rating', 'genres']]
print(top5_api)
print()

print("DELETE Dataset:")
top5_delete = df_delete.nlargest(5, 'avg_rating')[['title', 'avg_rating', 'genres']]
print(top5_delete)
print()

# Check if the same movies exist in all datasets
print("=== MOVIE AVAILABILITY COMPARISON ===\n")

nltk_movies = set(df_nltk['movieId'].unique())
api_movies = set(df_api['movieId'].unique())
delete_movies = set(df_delete['movieId'].unique())

print(f"Movies only in NLTK: {len(nltk_movies - api_movies - delete_movies)}")
print(f"Movies only in API: {len(api_movies - nltk_movies - delete_movies)}")
print(f"Movies only in DELETE: {len(delete_movies - nltk_movies - api_movies)}")
print(f"Movies in all three: {len(nltk_movies & api_movies & delete_movies)}")

# Check rating differences for same movies
print("\n=== RATING DIFFERENCES FOR SAME MOVIES ===\n")

# Find common movies
common_movies = nltk_movies & api_movies & delete_movies
print(f"Analyzing {len(common_movies)} common movies...")

# Sample a few movies to check rating differences
sample_movies = list(common_movies)[:10]
for movie_id in sample_movies:
    nltk_rating = df_nltk[df_nltk['movieId'] == movie_id]['avg_rating'].iloc[0]
    api_rating = df_api[df_api['movieId'] == movie_id]['avg_rating'].iloc[0]
    delete_rating = df_delete[df_delete['movieId'] == movie_id]['avg_rating'].iloc[0]
    
    title = df_nltk[df_nltk['movieId'] == movie_id]['title'].iloc[0]
    
    if not (nltk_rating == api_rating == delete_rating):
        print(f"Movie: {title}")
        print(f"  NLTK: {nltk_rating:.3f}, API: {api_rating:.3f}, DELETE: {delete_rating:.3f}")