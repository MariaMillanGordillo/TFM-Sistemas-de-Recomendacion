# 📚 TFM: Sistemas de Recomendación

Este repositorio contiene el desarrollo de mi Trabajo de Fin de Máster, centrado en la implementación y comparación de distintos sistemas de recomendación aplicados al dominio de libros.

🔗 **[Accede aquí a todos los archivos del proyecto (Drive)](https://drive.google.com/drive/folders/1SwLuxdjeabNd-NEt4kJ2-HfGH3yo3OFV?usp=share_link)**

⚠️ Debido a las limitaciones de tamaño de GitHub, algunos archivos no están subidos en este repositorio, pero están disponibles en el enlace anterior.

---

## 📂 Estructura del Proyecto

### 🔎 1. CBF – Filtrado Basado en Contenido

Modelo que utiliza información del contenido (título, géneros, autores) para recomendar libros similares.

* `CBF.ipynb`

---

### 👥 2. CF – Filtrado Colaborativo

Modelo basado en las interacciones usuario-libro, usando LightGCN.

* `CF.ipynb`
* `CF_hiperparametrizacion.ipynb`
* `lightgcn_CF_embeddings.pt`
* `lightgcn_CF_model.pth`

---

### 🔀 3. Hybrid – Sistemas Híbridos

Combinación de CBF y CF para mejorar las recomendaciones.

* `Híbrido_Aumento.ipynb` – Inicialización de embeddings de LigthGCN con información de contenido (CBF).
* `Híbrido_Combinación.ipynb` – Combinación de embeddings CBF y CF con una MLP.
* `HA_model.pth`
* `HA_embeddings.pt`
* `HC_model.pth`

---

### 📚 4. Dataset

Archivos procesados y originales del dataset de Goodreads.

* `goodreads_books_young_adult.json`
* `goodreads_book_authors.json.gz`
* `goodreads_book_genres_initial.json.gz`
* `interactions_filtered.csv` - Version filtrada de `goodreads_interactions.csv`.
* `user_id_map.csv`
* `book_id_map.csv`
* `books_authors_genres.csv`
* `sim_matrix.parquet`
* `tf-idf.parquet`

---

### 🌐 5. Streamlit App

Aplicación interactiva desarrollada con Streamlit para visualizar recomendaciones según el modelo elegido.

#### Archivos principales:

* `app.py`
* `func.py`

#### Páginas de la app:

* `pages/users.py`
* `pages/content.py`
* `pages/collaborative.py`
* `pages/hybrid.py`

#### Dataset para la app:

* `interactions_CF.csv`
* `user_id_map.csv`
* `book_id_map.csv`
* `books_authors_genres.csv`
* `sim_matrix.parquet`
* `tf-idf.parquet`

#### Modelos cargados:

* `lightgcn_CF_embeddings.pt`
* `HC_model.pth`

---

## 🛠 Requisitos

Este proyecto ha sido desarrollado en Python. Se recomienda usar un entorno virtual.

Puedes instalar las dependencias necesarias con alguna de las siguientes opciones:

- Usando pip y el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```
- Usando Conda y el archivo `environment.yml` para crear un entorno con todas las dependencias específicas:
```bash
conda env create -f environment.yml
conda activate nombre_del_entorno
```

---

## 💡 Créditos y Contacto

Trabajo desarrollado por **María Millán Gordillo**, como parte del Trabajo de Fin de Máster en Inteligencia Artificial.

