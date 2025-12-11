import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpersfunc import handle_duplicate_movies, encode_genres, concat_ratings_movies,drop_missing_genres,check_and_clean_consistency
movies_df = pd.read_csv(r'c:\Users\maata\Desktop\E-commerce\data\raw\movies.csv')
ratings_df = pd.read_csv(r'c:\Users\maata\Desktop\E-commerce\data\raw\ratings.csv')
df_del_cleaned = drop_missing_genres(movies_df)
df_del_cleaned = handle_duplicate_movies(df_del_cleaned)
ratings_df = check_and_clean_consistency(ratings_df, df_del_cleaned,  clean=True)
df_del_encoded = encode_genres(df_del_cleaned, column_name='genres')
print("\n--- Data After Cleaning and Encoding ---")
print(df_del_encoded.head(10))
full_df_del_encoded = concat_ratings_movies(df_del_encoded, ratings_df)
CLEAN_DATASET_EXP2_CSV = r'c:\Users\maata\Desktop\E-commerce\data\processed\clean_dataset_expDELETE.csv'
full_df_del_encoded.to_csv(CLEAN_DATASET_EXP2_CSV, index=False)