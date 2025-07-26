import os

# Tentukan direktori tempat FAISS index disimpan
VECTOR_STORE_DIR = os.getenv('VECTOR_STORE_DIR', 'vector_store')

# Tentukan nama file FAISS index yang ingin dihapus (index.faiss)
faiss_index_file = os.path.join(VECTOR_STORE_DIR, 'index.faiss')

# Periksa apakah file FAISS ada, lalu hapus
if os.path.exists(faiss_index_file):
    os.remove(faiss_index_file)
    print(f"FAISS index file '{faiss_index_file}' telah dihapus.")
else:
    print("FAISS index file tidak ditemukan.")
