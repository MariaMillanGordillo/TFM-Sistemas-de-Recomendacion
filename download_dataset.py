import requests
import os

# Lista de archivos a descargar
file_names = [
    'goodreads_books.json.gz',
    'goodreads_book_authors.json.gz',
    'goodreads_book_genres_initial.json.gz',
    'goodreads_interactions.csv',
    'book_id_map.csv',
    'user_id_map.csv',
    'goodreads_reviews_dedup.json.gz'
]

# Construir el diccionario de URLs de descarga
file_name_url = {
    fname: f'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/{fname}'
    for fname in file_names
}

# Función de descarga
def download_by_name(fname, local_filename):
    """Descarga un archivo dado su nombre y lo guarda en la ruta especificada."""
    url = file_name_url.get(fname)
    if url:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f'Dataset {fname} has been downloaded!')
    else:
        print(f'Dataset {fname} cannot be found!')

# Directorio de salida
OUT_DIR = '/Users/maria/Desktop/Máster IA/TFM/Code/Dataset'
os.makedirs(OUT_DIR, exist_ok=True)

# Descargar los archivos
for fname in file_names:
    output_path = os.path.join(OUT_DIR, fname)
    download_by_name(fname, output_path)
