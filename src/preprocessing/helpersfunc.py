import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# --- DUPLICATE HANDLING START ---
def handle_duplicate_movies(movies):
    # 1. Create a temporary column to count how many genres are listed
    # We split by '|' and count the length of the resulting list
    movies['genre_count'] = movies['genres'].apply(lambda x: len(str(x).split('|')))
    print(movies)
    # 2. Sort the dataframe
    # We sort by 'movieId' to group duplicates together, 
    # and by 'genre_count' in descending order (False) to put the one with MOST genres first.
    movies = movies.sort_values(by=['movieId', 'genre_count'], ascending=[True, False])

    # 3. Drop duplicates
    # We look for duplicates in 'movieId'. Since we sorted the "best" one to the top,
    # keep='first' will save the one with more genres and delete the others.
    movies = movies.drop_duplicates(subset='movieId', keep='first')

    # 4. Clean up: Remove the temporary helper column
    movies = movies.drop(columns=['genre_count'])
    return movies



def encode_genres(df, column_name='genres'):
    """
    Takes a dataframe and a column name with pipe-separated values.
    Returns the dataframe with new binary columns added.
    """
    
    # 2. The Magic: Split by '|' and create binary columns (0 or 1)
    # This automatically handles the "Action|Adventure" logic
    dummies = df[column_name].str.get_dummies(sep='|')
    
    # 3. Concatenate: Glue the original table and the new binary table together
    # axis=1 means we are adding columns next to each other, not rows
    df_encoded = pd.concat([df, dummies], axis=1)
    #df_encoded = df_encoded.drop(columns=[column_name])  # Optional: Drop it kif tabda fi model
    return df_encoded

# --- HOW TO USE IT ---
# Apply the function to your movies dataframe

def concat_ratings_movies(movies_encoded, ratings):
    # 3. Analyse exploratoire des données (EDA)

    # Calculer la note moyenne pour chaque film
    avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.columns = ['movieId', 'avg_rating']

    # Merge average ratings back into the movies dataframe so we have names and genres
    df_final = pd.merge(movies_encoded, avg_ratings, on='movieId', how='inner')

    # Identifier les 5 films les mieux notés
    top_5_movies = df_final.sort_values(by='avg_rating', ascending=False).head(5)
    print("\nTop 5 Rated Movies:")
    print(top_5_movies[['title', 'genres', 'avg_rating']])

    # Créer un graphique en barres (Distribution des évaluations)
    plt.figure(figsize=(10, 6))
    # We use the original ratings file for the distribution of ALL votes
    sns.countplot(x='rating', data=ratings, palette='viridis')

    # Annoter le graphique
    plt.title('Distribution des évaluations (Ratings)')
    plt.xlabel('Évaluation')
    plt.ylabel('Nombre')
    plt.show()
    return df_final


def drop_missing_genres(df):
    """
    Removes rows where genres is '(no genres listed)'
    Returns a new, clean DataFrame.
    """
    # 1. Define the placeholder to look for
    missing_label = '(no genres listed)'

    # 2. Count rows before
    initial_count = len(df)

    # 3. Create the clean dataframe (keep rows that are NOT the missing label)
    df_clean = df[df['genres'] != missing_label].copy()

    # 4. Count rows after
    final_count = len(df_clean)
    dropped_count = initial_count - final_count

    # 5. Print a report
    print(f"--- Cleanup Report ---")
    print(f"Initial rows: {initial_count}")
    print(f"Dropped rows: {dropped_count}")
    print(f"Final rows:   {final_count}")

    return df_clean


def check_and_clean_consistency(df_checking, df_reference, id_column='movieId', clean=False):
    """
    Checks if IDs in 'df_checking' exist in 'df_reference'.
    Optionally deletes the rows in 'df_checking' that don't match.

    Parameters:
    - df_checking: The DF to clean (e.g., ratings_df)
    - df_reference: The Source of Truth (e.g., movies_df)
    - clean: If True, returns a cleaned version of df_checking.

    Returns:
    - If clean=True: The cleaned DataFrame
    - If clean=False: The list of missing IDs
    """

    # 1. Get unique sets
    ids_in_checking = set(df_checking[id_column].unique())
    ids_in_reference = set(df_reference[id_column].unique())

    # 2. Find IDs in checking that represent "Dead Links"
    # (IDs that exist in ratings but NOT in movies)
    missing_ids = list(ids_in_checking - ids_in_reference)
    num_missing = len(missing_ids)

    # 3. Print Report
    print(f"### Consistency Check: {id_column} ###")
    print(f"Rows in checking DF: {len(df_checking)}")
    print(f"IDs in checking but NOT in reference: {num_missing}")

    if num_missing == 0:
        print("✅ Data is consistent. No rows need deletion.")
        if clean:
            return df_checking
        return missing_ids

    # 4. Cleaning Logic
    if clean:
        print(f"⚠️ Removing rows with {num_missing} invalid IDs...")

        # Keep only rows where the ID exists in the reference set
        df_clean = df_checking[df_checking[id_column].isin(ids_in_reference)]

        print(f"New row count: {len(df_clean)}")
        print(f"Dropped {len(df_checking) - len(df_clean)} rows.")
        return df_clean

    else:
        print(f"⚠️ Run with clean=True to remove these IDs.")
        return missing_ids