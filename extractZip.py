import zipfile
import os

zip_path = "imagenes_filtrada_gaussian_blur.zip"
extract_to = "imagenes_filtradas"

# Descomprimir
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

# Verificar ruta final
print("Imágenes extraídas en:", os.path.abspath(extract_to))
