import streamlit as st
import pandas as pd
import torch
from func import menu, load_dataframes, load_lightgcn_model, lightgcn_user_predictions, show_recommendations

def main():
    menu()
    st.title('📚 Collaborative Filtering Recommendations')

    books, _, _ = load_dataframes()

    # Get custom users
    custom_users = list(st.session_state.get('usuarios', {}).keys())

    if not custom_users:
        st.warning("No users found. Go to the Users page to create one.")
        return

    user_sel = st.selectbox('Select a user:', [''] + custom_users, key='collaborative_page_user')
    if user_sel != '':
        k = st.slider('Number of recommendations:', 1, 10, 5)

        # Favorite books selected by the custom user
        fav_books_titles = st.session_state['usuarios'][user_sel]
        fav_work_ids = books[books['title'].isin(fav_books_titles)]['work_id'].tolist()

        st.subheader(f"{user_sel}'s favorite books:")
        show_recommendations(fav_work_ids, books, k=len(fav_work_ids))

        # Load embeddings and mappings
        model_data = load_lightgcn_model()

        # Get recommendations in work_id format
        recommendations = lightgcn_user_predictions(
            user_sel=user_sel,
            books=books,
            k=k,
            model_data=model_data,
            fav_books_titles=fav_books_titles
        )

        st.subheader("Recommendations:")
        show_recommendations(recommendations, books, k=len(recommendations))

if __name__ == "__main__":
    main()
