import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Ruta donde están las imágenes filtradas
image_dir = "imagenes_filtradas/imagenes_filtrada_gaussian_blur"

# Cargar imágenes y etiquetas
X = []
y = []

for filename in os.listdir(image_dir):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        try:
            path = os.path.join(image_dir, filename)
            image = Image.open(path).convert('L').resize((64, 64))
            pixels = np.array(image).flatten()
            label = filename.split('_')[-1].split('.')[0]  # Extraer clase desde el nombre
            X.append(pixels)
            y.append(label)
        except Exception as e:
            print(f"No se pudo procesar {filename}: {e}")

print(f"Total imágenes cargadas: {len(X)}")

# Separar datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
print("\n=== Reporte de Clasificación ===")
print(classification_report(y_test, y_pred))

# Exportar a .arff para Weka
output_arff = "alzheimer_gaussian_filtered.arff"
with open(output_arff, 'w') as f:
    f.write("@RELATION alzheimer_gaussian_filtered\n\n")
    for i in range(len(X_train[0])):
        f.write(f"@ATTRIBUTE pixel{i} REAL\n")
    f.write("@ATTRIBUTE class {0,1,2,3}\n\n")
    f.write("@DATA\n")
    for xi, yi in zip(X + X_test, y + y_test):
        line = ",".join(map(str, xi)) + f",{yi}\n"
        f.write(line)

print(f"\nArchivo .arff exportado correctamente como: {output_arff}")
