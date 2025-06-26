import streamlit as st
import pandas as pd
import torch
from func import (
    menu,
    load_dataframes,
    load_hybrid_model,
    hybrid_user_predictions,
    show_recommendations,
    generate_user_embedding
)

def main():
    menu()
    st.title('📚 Hybrid Recommendations')

    books, sim_matrix, _ = load_dataframes()

    # Get custom users
    custom_users = list(st.session_state.get('usuarios', {}).keys())

    if not custom_users:
        st.warning("No users found. Go to the Users page to create one.")
        return

    user_sel = st.selectbox('Select a user:', [''] + custom_users, key='hybrid_page_user')
    if user_sel != '':
        k = st.slider('Number of recommendations:', 1, 10, 5)

        # Favorite books selected by the custom user
        fav_books_titles = st.session_state['usuarios'][user_sel]
        fav_work_ids = books[books['title'].isin(fav_books_titles)]['work_id'].tolist()

        st.subheader(f"{user_sel}'s favorite books:")
        show_recommendations(fav_work_ids, books, k=len(fav_work_ids))

        # Load hybrid model data (embeddings, mappings)
        model_data = load_hybrid_model()
        item_embeddings_gcn = model_data['item_embeddings']
        book_mapping = model_data['book_mapping']

        # Generate user embedding from favorite titles
        user_embedding_custom = generate_user_embedding(fav_books_titles, books, item_embeddings_gcn, book_mapping)

        # Get hybrid recommendations in work_id format
        recommendations = hybrid_user_predictions(
            user_embedding_custom=user_embedding_custom,
            books=books,
            k=k,
            fav_books_titles=fav_books_titles,
            model_data=model_data
        )

        st.subheader("Recommendations:")
        show_recommendations(recommendations, books, k=len(recommendations))

if __name__ == "__main__":
    main()
