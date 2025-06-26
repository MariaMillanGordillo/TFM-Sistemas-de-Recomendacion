import os
import torch
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(__file__)

cf_model_path = os.path.join(BASE_DIR, 'Models', 'lightgcn_CF_embeddings.pt')
tfidf_path = os.path.join(BASE_DIR, 'Dataset', 'tf-idf.parquet')
user_map_path = os.path.join(BASE_DIR, 'Dataset', 'user_id_map.csv')
book_map_path = os.path.join(BASE_DIR, 'Dataset', 'book_id_map.csv')
hc_model_path = os.path.join(BASE_DIR, 'Models', 'HC_model.pth')

def menu():
    with st.sidebar:
        st.page_link('app.py', label='Main Page', icon='🏠')
        st.page_link('pages/users.py', label='Users', icon='👤')
        st.page_link('pages/content.py', label='Content-Based Recommendations', icon='📚')
        st.page_link('pages/collaborative.py', label='Collaborative Filtering Recommendations', icon='📚')
        st.page_link('pages/hybrid.py', label='Hybrid Recommendations', icon='📚')

#### CARGAR DATASETS SOLO UNA VEZ ####
def load_dataframes():
    if ('books' not in st.session_state
        or 'sim_matrix' not in st.session_state
        or 'interactions' not in st.session_state):

        books = pd.read_csv(os.path.join(BASE_DIR, 'Dataset', 'books_authors_genres.csv'))
        sim_matrix = pd.read_parquet(os.path.join(BASE_DIR, 'Dataset', 'sim_matrix.parquet'))
        interactions = pd.read_csv(os.path.join(BASE_DIR, 'Dataset', 'interactions_CF.csv'))

        if sim_matrix.index.dtype == "object":
            sim_matrix.index = sim_matrix.index.astype(str)
        if sim_matrix.columns.dtype == "object":
            sim_matrix.columns = sim_matrix.columns.astype(str)
        
        st.session_state.books = books
        st.session_state.sim_matrix = sim_matrix
        st.session_state.interactions = interactions
    
    return st.session_state.books, st.session_state.sim_matrix, st.session_state.interactions


#### CONTENIDO BASED RECOMMENDATIONS ####
def find_nearest_neighbours(sim_matrix, work_id, k=5):
    similitudes = sim_matrix[work_id]
    indices = np.argsort(-similitudes)[1:k+1]
    return sim_matrix.index[indices].tolist()


#### CARGAR MODELO LIGHTGCN SOLO UNA VEZ ####
def load_lightgcn_model():
    if ('users_emb_final' in st.session_state
        and 'items_emb_final' in st.session_state
        and 'user_mapping' in st.session_state
        and 'book_mapping' in st.session_state):
        return (
            st.session_state.users_emb_final,
            st.session_state.items_emb_final,
            st.session_state.user_mapping,
            st.session_state.book_mapping
        )

    emb = torch.load(cf_model_path, map_location=torch.device('cpu'))

    user_map_df = pd.read_csv(user_map_path)
    user_mapping = dict(zip(user_map_df['user_id'], user_map_df['user_id_csv'].astype(int)))

    book_map_df = pd.read_csv(book_map_path)
    book_mapping = dict(zip(book_map_df['book_id'], book_map_df['book_id_csv'].astype(int)))

    st.session_state.users_emb_final = emb['users_emb_final_CF']
    st.session_state.items_emb_final = emb['items_emb_final_CF']
    st.session_state.user_mapping = user_mapping
    st.session_state.book_mapping = book_mapping

    return emb['users_emb_final_CF'], emb['items_emb_final_CF'], user_mapping, book_mapping


#### COLLABORATIVE FILTERING RECOMMENDATIONS ####
def lightgcn_user_predictions(user_sel, books, k, model_data, fav_books_titles):
    users_emb_final, items_emb_final, _, book_mapping = model_data

    book_id_to_index = {book_id: idx for book_id, idx in book_mapping.items()}

    fav_books_df = books[books['title'].isin(fav_books_titles)]
    fav_book_ids = fav_books_df['book_id'].tolist()

    fav_indices = [book_id_to_index[b] for b in fav_book_ids if b in book_id_to_index]

    if len(fav_indices) == 0:
        return []

    item_embs_fav = items_emb_final[fav_indices]
    user_emb_custom = torch.mean(item_embs_fav, dim=0, keepdim=True)

    scores = torch.matmul(user_emb_custom, items_emb_final.T).squeeze(0)
    scores_sorted, indices_sorted = torch.sort(scores, descending=True)

    selected_set = set(fav_book_ids)
    book_id_to_work_id = dict(zip(books['book_id'], books['work_id']))

    recommendations = []
    for idx in indices_sorted.tolist():
        book_id = {v: k for k, v in book_mapping.items()}.get(idx)
        if book_id is None or book_id in selected_set:
            continue

        work_id = book_id_to_work_id.get(book_id)
        if work_id is not None:
            recommendations.append(work_id)

        if len(recommendations) >= k:
            break

    return recommendations


#### MODELO HÍBRIDO ####
class HybridRecommenderNN(nn.Module):
    def __init__(self, gcn_dim, tfidf_dim, hidden_dims=[128, 64], dropout_rate=0.3):
        super().__init__()
        input_dim = gcn_dim + tfidf_dim
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.output = nn.Linear(hidden_dims[1], 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, gcn_emb, tfidf_emb):
        x = torch.cat([gcn_emb, tfidf_emb], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.output(x)
        return x.squeeze()


#### CARGAR MODELO HÍBRIDO SOLO UNA VEZ ####
def load_hybrid_model():
    if ('hybrid_model' in st.session_state
        and 'hybrid_item_embeddings' in st.session_state
        and 'hybrid_user_embeddings' in st.session_state
        and 'hybrid_tfidf_embeddings' in st.session_state
        and 'hybrid_book_mapping' in st.session_state):
        return {
            'model': st.session_state.hybrid_model,
            'item_embeddings': st.session_state.hybrid_item_embeddings,
            'user_embeddings': st.session_state.hybrid_user_embeddings,
            'tfidf_embeddings': st.session_state.hybrid_tfidf_embeddings,
            'book_mapping': st.session_state.hybrid_book_mapping,
        }

    gcn_data = torch.load(cf_model_path, map_location='cpu')
    item_embeddings_gcn = gcn_data['items_emb_final_CF']
    user_embeddings_gcn = gcn_data['users_emb_final_CF']

    tfidf_df = pd.read_parquet(tfidf_path, engine='pyarrow')
    tfidf_dim = 128
    pca = PCA(n_components=tfidf_dim)
    tfidf_tensor = torch.tensor(pca.fit_transform(tfidf_df), dtype=torch.float32)

    model = HybridRecommenderNN(gcn_dim=user_embeddings_gcn.shape[1], tfidf_dim=tfidf_tensor.shape[1])
    model.load_state_dict(torch.load(hc_model_path, map_location='cpu'))
    model.eval()

    book_map_df = pd.read_csv(book_map_path)
    book_mapping = dict(zip(book_map_df['book_id'], book_map_df['book_id_csv'].astype(int)))

    st.session_state.hybrid_model = model
    st.session_state.hybrid_item_embeddings = item_embeddings_gcn
    st.session_state.hybrid_user_embeddings = user_embeddings_gcn
    st.session_state.hybrid_tfidf_embeddings = tfidf_tensor
    st.session_state.hybrid_book_mapping = book_mapping

    return {
        'model': model,
        'item_embeddings': item_embeddings_gcn,
        'user_embeddings': user_embeddings_gcn,
        'tfidf_embeddings': tfidf_tensor,
        'book_mapping': book_mapping,
    }


#### GENERAR EMBEDDING PERSONALIZADO ####
def generate_user_embedding(user_fav_titles, books, item_embeddings_gcn, book_mapping):
    fav_book_ids = books[books['title'].isin(user_fav_titles)]['book_id'].tolist()
    embedding_list = []

    for book_id in fav_book_ids:
        if book_id in book_mapping:
            idx = book_mapping[book_id]
            embedding_list.append(item_embeddings_gcn[idx].numpy())

    if not embedding_list:
        raise ValueError("No se encontraron embeddings para los libros favoritos seleccionados.")

    avg_embedding = np.mean(embedding_list, axis=0)
    return torch.tensor(avg_embedding, dtype=torch.float32)


#### RECOMENDACIONES HÍBRIDAS ####
@torch.no_grad()
def hybrid_user_predictions(user_embedding_custom, books, k, fav_books_titles, model_data):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fav_book_ids = books[books['title'].isin(fav_books_titles)]['book_id'].tolist()
    user_seen_books = set(fav_book_ids)

    model = model_data['model'].to(device)
    item_embeddings_gcn = model_data['item_embeddings'].to(device)
    tfidf_item_embeddings = model_data['tfidf_embeddings'].to(device)
    book_mapping = model_data['book_mapping']

    user_embedding = user_embedding_custom.to(device)

    scores = []
    candidates = []

    book_map_df = books[['book_id', 'work_id']].copy()
    book_map_df['idx'] = book_map_df['book_id'].map(book_mapping)

    for _, row in book_map_df.iterrows():
        book_id = row['book_id']
        work_id = row['work_id']
        idx = row['idx']

        if book_id in user_seen_books or pd.isna(idx):
            continue
        idx = int(idx)
        if idx >= item_embeddings_gcn.shape[0] or idx >= tfidf_item_embeddings.shape[0]:
            continue

        gcn_item = item_embeddings_gcn[idx]
        tfidf_item = tfidf_item_embeddings[idx]
        gcn_comb = user_embedding * gcn_item

        score = torch.sigmoid(model(gcn_comb.unsqueeze(0), tfidf_item.unsqueeze(0))).item()

        scores.append(score)
        candidates.append(work_id)

    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    top_k_work_ids = [candidates[i] for i in top_k_indices]

    return top_k_work_ids


#### SHOW RECOMMENDATIONS ####
def show_recommendations(rec_books, books, k):
    num_columns = 3 if k <= 6 or k % 3 == 0 else 4
    rows = [rec_books[i:i + num_columns] for i in range(0, len(rec_books), num_columns)]

    for row in rows:
        cols = st.columns(num_columns)

        for i, rec_id in enumerate(row):
            rec_book = books[books['work_id'].astype(str) == str(rec_id)]
            
            title = rec_book['title'].values[0]
            image_url = rec_book['image_url'].values[0]

            with cols[i]:
                image_container = st.container(height=285, border=False)  # Contenedor para la imagen
                popover_container = st.container(height=100, border=False)  # Contenedor para el popover
                
                with image_container:
                    st.image(image_url, caption=title, width=143)

                with popover_container:
                    with st.popover("See details"):
                        st.write(f"**Title:** {title}")
                        st.write(f"**Author:** {rec_book['author_name'].values[0]}")
                        st.write(f"**Rating:** {rec_book['average_rating'].values[0]}")

                        num_pages = rec_book['num_pages'].values[0]
                        st.write(f"**Number of pages:** {str(int(num_pages)) if pd.notna(num_pages) else 'Unknown'}")

                        genres = rec_book['genres'].values[0]
                        if pd.notna(genres):
                            genres = genres.replace('[', '').replace(']', '').replace('\'', '')
                            genres = genres.split(', ')
                            st.write(f"**Genres:** {', '.join(genres)}.")
                        else:
                            st.write("**Genres:** Unknown")

                        description = rec_book['description'].values[0]
                        st.write(f"**Description:** {description if pd.notna(description) else 'Unknown'}")

