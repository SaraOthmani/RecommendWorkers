import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Get the path to the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# File paths for workers and shifts data
workers_file_path = os.path.join(current_directory, 'db_workers.xlsx')
shifts_file_path = os.path.join(current_directory, 'db_shifts.xlsx')

df_workers = pd.read_excel(workers_file_path)
df_shifts = pd.read_excel(shifts_file_path)

# File paths for images
human_image_path = os.path.join(current_directory, 'human.png')

#treating and pre-processing data
df_workers['evaluation_score'] = df_workers['evaluation_score'].round(2)

#adding a column based on the years of experience to the workers database
# Define a function to apply the mapping logic
def experience_category(years):
    if years == 0:
        return "Aucune"
    elif 1 <= years <= 4:
        return "Moyenne"
    else:
        return "Forte"

# Create a new column 'experience' based on 'annee_experience' column
df_workers['experience'] = df_workers['annee_experience'].apply(experience_category)

#adding a column based on the years of required experience to the shifts database
def experience_category2(required_years):
    if required_years == 0:
        return "Aucune"
    elif 1 <= required_years <= 4:
        return "Moyenne"
    else:
        return "Forte"

df_shifts['niveau_experience'] = df_shifts['experience_requise'].apply(experience_category2)

#Relevant features/criteria column for the workers dataset
df_workers['workers_features'] = df_workers['poste_occupe'] + " " + df_workers['secteur_activite'] + " " + df_workers['experience']

#Relevant features/criteria column for the shifts dataset
df_shifts['posts_features'] = df_shifts['nom_poste'] + " " + df_shifts['secteur'] + " " + df_shifts['niveau_experience']

# Combine 'workers_features' and 'posts_features' text for CountVectorizer
combined_text = df_workers['workers_features'].tolist() + df_shifts['posts_features'].tolist()

# Apply CountVectorizer to transform combined text features into numerical vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vector_combined = cv.fit_transform(combined_text).toarray()

# Split the vectors back to workers and shifts vectors
vector_workers = vector_combined[:len(df_workers)]
vector_shifts = vector_combined[len(df_workers):]

# Calculate cosine similarity between workers and shifts
similarity_matrix = cosine_similarity(vector_workers, vector_shifts)

# Define a function to get recommended workers for a given post name
def get_recommended_workers(post_name):
    if post_name not in df_shifts['nom_poste'].values:
        st.error(f"Post name '{post_name}' not found in the database.")
        return pd.DataFrame()

    post_idx = df_shifts[df_shifts['nom_poste'] == post_name].index[0]
    similarity_scores = similarity_matrix[:, post_idx]
    top_workers_indices = similarity_scores.argsort()[::-1]

    # Filter out rows with similarity score of 0
    non_zero_indices = similarity_scores[top_workers_indices] != 0
    top_workers_indices = top_workers_indices[non_zero_indices]

    top_workers = df_workers.loc[top_workers_indices, ['id_worker', 'nom', 'poste_occupe', 'secteur_activite', 'experience']]
    top_workers['similarity_score'] = similarity_scores[top_workers_indices]
    return top_workers

# Set page layout to wide
st.set_page_config(layout="wide")

# Display the app title and image
st.title('Recommandation des travailleurs')
st.image(human_image_path, width=100)

# Display the description
st.write('À la recherche de freelancers ? GoodJob vous propose le travailleur idéal répondant à toutes vos exigences !')

# Display the list of available post names and get user selection
post_names = df_shifts['nom_poste'].unique()
selected_post_name = st.selectbox('Sélectionner un poste :', post_names)

# Filtering options
st.subheader('Options de filtrage')

# Similarity score range filter
min_similarity, max_similarity = st.slider('Plage de similarité', 0.0, 1.0, (0.5, 1.0))

# Evaluation score range filter
min_evaluation, max_evaluation = st.slider('Score d\'évaluation', 0.0, 5.0, (0.0, 5.0))

# Experience level filter
experience_levels = df_workers['experience'].unique()
selected_experience = st.multiselect('Niveau d\'expérience', experience_levels)

# Sector filter
sectors = df_workers['secteur_activite'].unique()
selected_sector = st.multiselect('Secteur d\'activité', sectors)

# Get recommended workers for the selected post name
recommended_workers_df = get_recommended_workers(selected_post_name)

if not recommended_workers_df.empty:
    # Map evaluation_score from df_workers to recommended_workers_df based on id_worker
    worker_scores = df_workers[['id_worker', 'evaluation_score']].set_index('id_worker')['evaluation_score']
    recommended_workers_df['evaluation_score'] = recommended_workers_df['id_worker'].map(worker_scores)

    # Classify workers based on their 'evaluation_score'
    score_bins = [0, 2.99, 3.99, 5]
    score_labels = ['à reconsidérer', 'recommandé', 'Fortement recommandé']
    recommended_workers_df['classification'] = pd.cut(recommended_workers_df['evaluation_score'], bins=score_bins, labels=score_labels)



    # Apply filters
    filtered_workers_df = recommended_workers_df[
        (recommended_workers_df['similarity_score'] >= min_similarity)
        & (recommended_workers_df['similarity_score'] <= max_similarity)
        & (recommended_workers_df['evaluation_score'] >= min_evaluation)
        & (recommended_workers_df['evaluation_score'] <= max_evaluation)
        & (recommended_workers_df['experience'].isin(selected_experience))
        & (recommended_workers_df['secteur_activite'].isin(selected_sector))
    ]
    
    if selected_experience:
        filtered_workers_df = filtered_workers_df[filtered_workers_df['experience'].isin(selected_experience)]
        
    if selected_sector:
        filtered_workers_df = filtered_workers_df[filtered_workers_df['secteur_activite'].isin(selected_sector)]

    # Drop 'secteur_activite' and 'experience' columns from the DataFrame
    filtered_workers_df = filtered_workers_df.drop(['secteur_activite', 'experience', 'similarity_score', 'evaluation_score'], axis=1)

    # Display the recommended workers
    st.subheader('On vous recommande:')
    st.dataframe(filtered_workers_df, height=500, width=700)
