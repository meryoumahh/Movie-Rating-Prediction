import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpersfunc import handle_duplicate_movies, encode_genres, concat_ratings_movies,drop_missing_genres,check_and_clean_consistency
movies_df = pd.read_csv(r'c:\Users\maata\Desktop\E-commerce\data\raw\movies.csv')
ratings_df = pd.read_csv(r'c:\Users\maata\Desktop\E-commerce\data\raw\ratings.csv')

# 1. Setup the Classifier
print("Loading Model... (this might take a minute)")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", framework="pt")

# UPDATED: Complete list of genres based on your data (added IMAX)
candidate_labels = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]


# 2. Create a copy of the dataframe to work on
df_nlp = movies_df.copy()

# 3. Define the prediction function
def predict_genre_from_title(row):
    # Only predict if the genre is missing
    if row['genres'] == '(no genres listed)':
        title = row['title']

        # Run the classifier
        result = classifier(title, candidate_labels, multi_label=True)

        # LOGIC: Take the top 3 predictions
        top_3_genres = result['labels'][:3]

        # Format as "Genre|Genre|Genre"
        predicted_genre_str = "|".join(top_3_genres)

        # Print guess above the progress bar
        tqdm.write(f"Movie: '{title}'  ->  Guessed: {predicted_genre_str}")

        return predicted_genre_str

    # If not missing, return the original genre
    return row['genres']

# 4. Apply the function
mask = df_nlp['genres'] == '(no genres listed)'
rows_to_fix = df_nlp[mask]

print(f"Predicting genres for {len(rows_to_fix)} movies...")

# Initialize tqdm for pandas
tqdm.pandas()
df_nlp.loc[mask, 'genres'] = df_nlp.loc[mask].progress_apply(predict_genre_from_title, axis=1)

# 5. Display the results
print("\n--- Final Data Check ---")
print(df_nlp.loc[mask].head(10))

# 6. Handle duplicates
df_nlp_cleaned = handle_duplicate_movies(df_nlp)
df_nlp_cleaned = drop_missing_genres(df_nlp_cleaned)
ratings_df = check_and_clean_consistency(ratings_df,df_nlp_cleaned, clean=True)
df_nlp_encoded = encode_genres(df_nlp_cleaned, column_name='genres')
print("\n--- Data After Cleaning and Encoding ---")
print(df_nlp_encoded.head(10))
df_final = concat_ratings_movies(df_nlp_encoded, ratings_df)
CLEAN_DATASET_EXP3_CSV = r'c:\Users\maata\Desktop\E-commerce\data\processed\clean_dataset_expNLTK.csv'
df_final.to_csv(CLEAN_DATASET_EXP3_CSV, index=False)
