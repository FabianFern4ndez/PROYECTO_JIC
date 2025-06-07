import pandas as pd
import cv2
import numpy as np
from pathlib import Path

def process_with_median(df):
    processed_data = []
    for index, row in df.iterrows():
        try:
            img_bytes = row['image']['bytes']
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("No se pudo decodificar la imagen")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Aplicar filtro de mediana (kernel 3x3)
            processed_img = cv2.medianBlur(gray, 3)
            
            _, processed_img_bytes = cv2.imencode('.png', processed_img)
            
            processed_data.append({
                'image': {'bytes': processed_img_bytes.tobytes()},
                'label': row['label']
            })
            
        except Exception as e:
            print(f"Error procesando fila {index}: {str(e)}")
            processed_data.append(row.to_dict())
    
    return pd.DataFrame(processed_data)

# Uso:
parquet_path = Path(r'C:\Users\radia\Downloads\archive\Alzheimer MRI Disease Classification Dataset\Data\train-00000-of-00001-c08a401c53fe5312.parquet')
df = pd.read_parquet(parquet_path)
df_median = process_with_median(df)
df_median.to_parquet(Path(r'C:\Users\radia\Downloads\archive\median_processed.parquet'))