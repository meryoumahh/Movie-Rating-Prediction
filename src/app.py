import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing.ApiExperiment.dataprepExp1 import API_genre_exp
# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="MovieLens Predictor", layout="wide")

st.title("🎬 Movie Ratings Prediction App")
st.markdown("""
**Project:** End-to-End Machine Learning Cycle  
**Goal:** Predict movie ratings based on Genres using Random Forest  
**Best Model Performance:** RMSE ~0.80
""")

# --- 2. DATA LOADING & PROCESSING (Cached for speed) ---
@st.cache_data
def load_data():
    # Load processed data from API experiment
    df_final = API_genre_exp()
    
    # Extract genre columns (all columns except movieId, title, genres, avg_rating)
    genre_columns = [col for col in df_final.columns if col not in ['movieId', 'title', 'genres', 'avg_rating']]
    
    return genre_columns, df_final

# Run the function
genre_list, df_final = load_data()
st.success(f"✅ Data Successfully Loaded! ({df_final.shape[0]} movies)")

# --- 3. TABS FOR ORGANIZATION ---
tab1, tab2, tab3 = st.tabs(["📊 EDA & Insights", "🧠 Model Training", "🔮 Live Prediction"])

# --- TAB 1: EDA ---
with tab1:
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution of Ratings")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df_final['avg_rating'], bins=20, kde=True, color='skyblue', ax=ax)
        ax.set_title("How are movies rated?")
        st.pyplot(fig)
        st.caption("Most movies are rated between 3.0 and 4.0.")

    with col2:
        st.subheader("Correlation: Genres vs Rating")
        # Let's show which genres correlate most with high ratings
        # We drop non-numeric columns for correlation
        corr = df_final.drop(['movieId', 'title', 'genres'], axis=1).corr()['avg_rating'].sort_values()
        # Remove the self-correlation
        corr = corr.drop('avg_rating')
        
        # Plot top 5 positive and negative correlations
        top_corr = pd.concat([corr.head(5), corr.tail(5)])
        
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        top_corr.plot(kind='barh', color='purple', ax=ax2)
        ax2.set_title("Genres with Highest/Lowest Correlation")
        st.pyplot(fig2)

# --- TAB 2: MODEL TRAINING ---
with tab2:
    st.header("Train the Model")
    st.write("We use **RandomForestRegressor** with parameters optimized via GridSearchCV.")
    
    if st.button("🚀 Train Model Now"):
        with st.spinner("Training Model... Please wait."):
            
            # 1. Prepare Features
            X = df_final.drop(['movieId', 'title', 'genres', 'avg_rating'], axis=1)
            y = df_final['avg_rating']
            
            # 2. Split Data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 3. Import and use the trained model from randomforest.py
            from model.randomforest import rf_model as trained_model
            model = trained_model
            
            # 4. Evaluate
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Save to session state so we can use it in Tab 3
            st.session_state['model'] = model
            st.session_state['rmse'] = rmse
            st.session_state['r2'] = r2
            
            # Display Metrics
            c1, c2 = st.columns(2)
            c1.metric("RMSE Error", f"{rmse:.4f}", help="Lower is better")
            c2.metric("R² Score", f"{r2:.2%}", help="Higher is better")
            
            # Feature Importance Plot
            st.subheader("What drives the prediction?")
            importances = model.feature_importances_
            feature_df = pd.DataFrame({'Genre': X.columns, 'Importance': importances})
            feature_df = feature_df.sort_values(by='Importance', ascending=False).head(10)
            
            fig3, ax3 = plt.subplots()
            sns.barplot(x='Importance', y='Genre', data=feature_df, palette='viridis', ax=ax3)
            st.pyplot(fig3)

# --- TAB 3: PREDICTION ---
with tab3:
    st.header("Predict a Movie Rating")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Please go to the 'Model Training' tab and click Train first!")
    else:
        st.write("Create a hypothetical movie by selecting genres:")
        
        # User selection
        selected_genres = st.multiselect("Select Genres:", genre_list)
        
        if st.button("Predict Rating"):
            if not selected_genres:
                st.error("Please select at least one genre.")
            else:
                # Create input vector (all zeros initially)
                input_data = pd.DataFrame(0, index=[0], columns=genre_list)
                
                # Set selected genres to 1
                for g in selected_genres:
                    input_data[g] = 1
                
                # Predict
                prediction = st.session_state['model'].predict(input_data)[0]
                
                st.markdown(f"### 🎬 Predicted Rating: **{prediction:.2f} / 5.0**")
                
                # Fun feedback
                if prediction > 3.5:
                    st.success("The model thinks this will be a hit!")
                elif prediction > 2.5:
                    st.info("The model thinks this will be an average movie.")
                else:
                    st.error("The model thinks this might be a flop.")