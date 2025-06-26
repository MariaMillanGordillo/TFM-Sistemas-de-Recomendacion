import streamlit as st
from func import menu, show_recommendations

def main():
    st.title("Create user and select favorite books")
    st.write("This page allows you to create a user and select your favorite books from the dataset. Once you create a user, you can see recommendations based on your favorite books.")
    st.write("Please select your favorite books and enter your username.")

    # Asegurarnos que los datos están cargados
    if 'books' not in st.session_state:
        st.warning("No books data found. Please load the dataset first.")
        return
    
    books = st.session_state.books

    # Mostrar solo títulos para seleccionar
    libros = books['title'].tolist()

    # Multiselección de libros
    libros_seleccionados = st.multiselect(
        "Select your favorite books",
        options=libros
    )

    # Input para nombre de usuario
    nombre_usuario = st.text_input("Enter your username", key='username_input')

    if st.button("Save user"):
        if not nombre_usuario:
            st.warning("Please enter a username.")
        elif len(libros_seleccionados) == 0:
            st.warning("Please select at least one book.")
        else:
            # Guardar usuario con libros favoritos en session_state
            if 'usuarios' not in st.session_state:
                st.session_state['usuarios'] = {}

            st.session_state.usuarios[nombre_usuario] = libros_seleccionados
            st.success(f"User '{nombre_usuario}' saved with {len(libros_seleccionados)} favorite books.")

    # Mostrar usuarios guardados
    if 'usuarios' in st.session_state and st.session_state.usuarios:
        st.subheader("Saved Users and their favorite books")
        for user, favs in st.session_state.usuarios.items():
            fav_work_ids = books[books['title'].isin(favs)]['work_id'].tolist()
            st.write(f"{user}'s favorite books:")
            show_recommendations(fav_work_ids, books, k=len(fav_work_ids))

if __name__ == "__main__":
    menu()
    main()