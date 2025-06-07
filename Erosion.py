import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import os

# 1. Cargar el archivo Parquet original
parquet_path = Path(r'c:\Users\radia\Downloads\archive\Alzheimer MRI Disease Classification Dataset\Data\train-00000-of-00001-c08a401c53fe5312.parquet')
df = pd.read_parquet(parquet_path)

# 2. Lista para almacenar las imágenes procesadas
processed_data = []

# 3. Definir el kernel para la erosión (tamaño y forma)
kernel = np.ones((3,3), np.uint8)  # Puedes ajustar el tamaño (5,5) para mayor efecto

# 4. Procesar cada imagen con erosión
for index, row in df.iterrows():
    try:
        # Extraer bytes de la imagen original
        img_bytes = row['image']['bytes']
        
        # Convertir a array numpy y decodificar
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen")
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar erosión (el filtro principal)
        eroded_img = cv2.erode(gray, kernel, iterations=1)  # iterations controla la intensidad
        
        # Convertir la imagen procesada a bytes (formato PNG)
        _, processed_img_bytes = cv2.imencode('.png', eroded_img)
        
        # Agregar a la lista de datos procesados
        processed_data.append({
            'image': {
                'bytes': processed_img_bytes.tobytes()
            },
            'label': row['label']
        })
        
    except Exception as e:
        print(f"Error procesando fila {index}: {str(e)}")
        # Si hay error, conservamos los datos originales
        processed_data.append(row.to_dict())
        continue

# 5. Crear DataFrame con los datos procesados
df_processed = pd.DataFrame(processed_data)

# 6. Guardar el nuevo Parquet
output_parquet_path = Path(r'C:\Users\radia\Downloads\archive\eroded_images.parquet')
df_processed.to_parquet(output_parquet_path)

print(f"Procesamiento completado. Imágenes con erosión guardadas en: {output_parquet_path}")