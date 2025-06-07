import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from skimage.restoration import denoise_nl_means
from skimage import exposure

def advanced_mri_processing(img):
    """
    Pipeline avanzado para procesamiento de imágenes MRI de Alzheimer
    Combina: Denoising no-local means + CLAHE + Unsharp Masking + Optimización de histograma
    """
    # 1. Conversión a escala de grises y normalización
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 2. Denoising no-local means (mejor que mediana/gaussiano para MRI)
    denoised = denoise_nl_means(gray, h=0.8, fast_mode=True, patch_size=5, patch_distance=3)
    denoised = (denoised * 255).astype(np.uint8)
    
    # 3. CLAHE adaptativo con parámetros optimizados para neuroimágenes
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10,10))
    clahe_img = clahe.apply(denoised)
    
    # 4. Unsharp masking para realzar bordes anatómicos
    blurred = cv2.GaussianBlur(clahe_img, (0,0), 3)
    sharpened = cv2.addWeighted(clahe_img, 1.7, blurred, -0.6, 0)
    
    # 5. Ecualización de histograma global para maximizar contraste
    hist_eq = exposure.equalize_hist(sharpened)
    final_img = (hist_eq * 255).astype(np.uint8)
    
    # 6. Apertura morfológica para eliminar artefactos residuales
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    processed = cv2.morphologyEx(final_img, cv2.MORPH_OPEN, kernel)
    
    return processed

def process_dataset(df):
    processed_data = []
    
    for index, row in df.iterrows():
        try:
            img_bytes = row['image']['bytes']
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Error decodificando imagen")
                
            # Procesamiento avanzado
            processed_img = advanced_mri_processing(img)
            
            # Codificación como PNG sin pérdida
            _, img_bytes = cv2.imencode('.png', processed_img)
            
            processed_data.append({
                'image': {'bytes': img_bytes.tobytes()},
                'label': row['label'],
                'original_shape': str(img.shape),
                'processed_shape': str(processed_img.shape)
            })
            
        except Exception as e:
            print(f"Error en fila {index}: {str(e)}")
            processed_data.append(row.to_dict())
    
    return pd.DataFrame(processed_data)

# Ejecución
parquet_path = Path(r'C:\Users\radia\Downloads\archive\Alzheimer MRI Disease Classification Dataset\Data\train-00000-of-00001-c08a401c53fe5312.parquet')
df = pd.read_parquet(parquet_path)
df_processed = process_dataset(df)

# Guardado
output_path = Path(r'C:\Users\radia\Downloads\archive\advanced_processed.parquet')
df_processed.to_parquet(output_path)
print(f"Procesamiento completado. Dataset guardado en: {output_path}")