import streamlit as st
import pandas as pd
import numpy as np
from func import menu, load_dataframes, find_nearest_neighbours, show_recommendations

def cbf_results():
    books, sim_matrix, _ = load_dataframes()

    st.title('Content-Based Recommendations')

    tabs = st.tabs(['By Book', 'By User'])

    # Tab 1: By Book
    with tabs[0]:
        st.subheader('Book finder')
        book_selection = st.selectbox('Select a book:', [""] + books['title'].tolist(), key='cbf_book_select')

        if book_selection != "":
            book = books[books['title'] == book_selection]
            col1, col2 = st.columns(2)
            with col1:
                st.write('#####')
                container1 = st.container()
                container1.write(f"\nTitle: {book['title'].values[0]}")
                container1.write(f"Author: {book['author_name'].values[0]}")
                container1.write(f"Rating: {book['average_rating'].values[0]}")
                num_pages = book['num_pages'].values[0]
                container1.write(f"Number of pages: {str(int(num_pages)) if pd.notna(num_pages) else 'Unknown'}")
                genres = book['genres'].values[0]
                if pd.notna(genres):
                    genres = genres.replace('[', '').replace(']', '').replace('\'', '')
                    genres = genres.split(', ')
                    container1.write(f"Genres: {', '.join(genres)}.")
                else:
                    container1.write("Genres: Unknown")
            with col2:
                container2 = st.container()
                container2.markdown(
                    f"<div style='display: flex; justify-content: center; align-items: center; height: 100%;'><img src='{book['image_url'].values[0]}' width='143'></div>",
                    unsafe_allow_html=True
                )

            k = st.slider('Number of recommendations:', 1, 10, 5, key='cbf_k_slider_book')
            rec_but = st.button('Get recommendations')

            if rec_but:
                work_id = book.iloc[0]['work_id']
                rec_books = find_nearest_neighbours(sim_matrix, work_id, k)

                st.subheader('Recommendations')
                show_recommendations(rec_books, books, k)

    # Tab 2: By User
    with tabs[1]:
        custom_users = list(st.session_state.get('usuarios', {}).keys())
        if not custom_users:
            st.warning("No custom users created. Go to Users page to create one.")
            return

        user_sel = st.selectbox('Select a user:', [''] + custom_users, key='cbf_user_select')
        if user_sel != '':
            k = st.slider('Number of recommendations:', 1, 10, 5, key='cbf_k_slider_user')

            fav_books_titles = st.session_state['usuarios'][user_sel]

            if not fav_books_titles:
                st.warning("Selected user has no favorite books.")
                return

            fav_work_ids = books[books['title'].isin(fav_books_titles)]['work_id'].tolist()

            st.subheader(f"{user_sel}'s favorite books:")
            show_recommendations(fav_work_ids, books, k=len(fav_work_ids))

            work_id_to_idx = {wid: idx for idx, wid in enumerate(books['work_id'])}
            fav_indices = [work_id_to_idx[w] for w in fav_work_ids if w in work_id_to_idx]

            if not fav_indices:
                st.warning("Favorite books not found in the similarity matrix.")
                return

            user_profile = np.mean(sim_matrix.iloc[fav_indices, :], axis=0)

            from sklearn.metrics.pairwise import cosine_similarity
            user_profile = user_profile.to_numpy().reshape(1, -1)
            sims = cosine_similarity(user_profile, sim_matrix.to_numpy())[0]

            sorted_indices = sims.argsort()[::-1]

            fav_set = set(fav_work_ids)
            recommendations = []
            for idx in sorted_indices:
                work_id = books.iloc[idx]['work_id']
                if work_id not in fav_set:
                    recommendations.append(work_id)
                if len(recommendations) >= k:
                    break

            st.subheader("Recommendations based on user's favorites:")
            show_recommendations(recommendations, books, k=len(recommendations))


def main():
    menu()
    cbf_results()

if __name__ == "__main__":
    main()
