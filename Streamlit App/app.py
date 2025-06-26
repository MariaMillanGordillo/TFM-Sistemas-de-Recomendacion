import streamlit as st
import pandas as pd
import numpy as np
import os
from func import menu, load_dataframes

def main():
    books, _, _ = load_dataframes()
    
    menu()
    
    st.title('Book Recommendation System')
    st.text('This website is a book recommendation system based on the GoodReads dataset.')
    
    st.subheader('Book finder')
    book_selection = st.selectbox('Select a book:', [""] + books['title'].tolist(), key='main_page_book')
    
    if book_selection != "":
        book = books[books['title'] == book_selection]

        col1, col2 = st.columns(2)
        with col1:
            st.write('#####')
            container1 = st.container(border=False, height=300)
            container1.write(f"\nTitle: {book['title'].values[0]}")
            container1.write(f"Author: {book['author_name'].values[0]}")
            container1.write(f"Rating: {book['average_rating'].values[0]}")
            num_pages = book['num_pages'].values[0]
            container1.write(f"Number of pages: {str(int(num_pages)) if pd.notna(num_pages) else 'Unknown'}")
            publisher = book['publisher'].values[0]
            container1.write(f"Publisher: {publisher if pd.notna(publisher) else 'Unknown'}")
            publication_day = book['publication_day'].values[0]
            publication_month = book['publication_month'].values[0]
            publication_year = book['publication_year'].values[0]
            if pd.notna(publication_day) and pd.notna(publication_month) and pd.notna(publication_year):
                container1.write(f"Publication date: {int(publication_day)}/{int(publication_month)}/{int(publication_year)}")
            genres = book['genres'].values[0]
            if pd.notna(genres):
                genres = genres.replace('[', '').replace(']', '').replace('\'', '')
                genres = genres.split(', ')
                container1.write(f"Genres: {', '.join(genres)}.")
            else:
                container1.write("Genres: Unknown")
        with col2:
            st.write('######')
            st.markdown(
                """
                <style>
                .image-container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 270px;  /* Altura fija */
                    width: 100%;  /* Ancho completo */
                    border: 1px solid #ccc;
                    border-radius: 10px;
                    padding: 10px;
                    background-color: #f9f9f9;
                    overflow: hidden;  /* Evita desbordamientos */
                }
                .image-container img {
                    height: 100%;  /* Ajusta la altura de la imagen al contenedor */
                    width: auto;  /* Mantiene la proporción */
                    max-width: 100%;  /* No supera el ancho del contenedor */
                    object-fit: contain;  /* Ajusta la imagen sin recortarla */
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="image-container">
                    <img src="{book['image_url'].values[0]}">
                </div>
                """,
                unsafe_allow_html=True
            )

        description = book['description'].values[0]
        st.write(f"Description: {description if pd.notna(description) else 'Unknown'}")

if __name__ == "__main__":
    main()
