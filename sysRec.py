import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

#importing the workers dataset
df_workers = pd.read_excel(r'C:\Users\user\OneDrive\Bureau\PFE\db_workers.xlsx')

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

#importing the workers dataset
df_shifts = pd.read_excel(r'C:\Users\user\OneDrive\Bureau\PFE\db_shifts.xlsx')

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
df_workers['workers_features'] = df_workers['poste_occupe'] + " " + df_workers['secteur_activite']+ " " + df_workers['experience']

#Relevant features/criteria column for the shifts dataset
df_shifts['posts_features'] = df_shifts['nom_poste'] + " " + df_shifts['secteur']+ " " + df_shifts['niveau_experience'] 

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

recommended_workers = []

for i, shift in df_shifts.iterrows():
    shift_id = shift['id_shift']
    post_name = shift['nom_poste']
    similarity_scores = similarity_matrix[:, i]  # Get similarity scores for the current shift
    top_workers_indices = np.argsort(similarity_scores)[::-1]  # Get indices of all similar workers
    top_workers = df_workers.loc[top_workers_indices, ['id_worker', 'nom']]  # Get worker details
    top_workers['similarity_score'] = similarity_scores[top_workers_indices]  # Add similarity score column
    top_workers['shift_id'] = shift_id  # Add shift_id column
    top_workers['nom_poste'] = post_name  # Add post_name column
    recommended_workers.append(top_workers)

recommended_workers_df = pd.concat(recommended_workers)
recommended_workers_df.reset_index(drop=True, inplace=True)

# Map evaluation_score from df_workers to recommended_workers_df based on id_worker
worker_scores = df_workers[['id_worker', 'evaluation_score']].set_index('id_worker')['evaluation_score']
recommended_workers_df['evaluation_score'] = recommended_workers_df['id_worker'].map(worker_scores)

# Classify workers based on their 'evaluation_score'
score_bins = [0, 2.99, 3.99, 5]
score_labels = ['à reconsidérer', 'recommandé', 'Fortement recommandé']
recommended_workers_df['classification'] = pd.cut(recommended_workers_df['evaluation_score'], bins=score_bins, labels=score_labels)