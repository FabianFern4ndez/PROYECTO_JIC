import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Ruta a imágenes
image_dir = "imagenes_filtradas/imagenes_filtrada_gaussian_blur"

# Diccionario clases
label_map = {
    '0': 'sinDemencia',
    '1': 'leveDemencia',
    '2': 'mediaDemencia',
    '3': 'moderadaDemencia'
}

# Carga y preprocesamiento
X = []
y = []

for fname in os.listdir(image_dir):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            label_num = fname.split('_')[-1].split('.')[0]
            label = label_map.get(label_num)
            if label:
                img_path = os.path.join(image_dir, fname)
                img = Image.open(img_path).convert("L").resize((128, 128))
                X.append(np.array(img) / 255.0)  # Normalizar
                y.append(label)
        except Exception as e:
            print(f"Error en {fname}: {e}")

# Convertir a arrays
X = np.array(X).reshape(-1, 128, 128, 1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Separar datos
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluar modelo
y_pred = model.predict(X_test)
y_pred_labels = le.inverse_transform(np.argmax(y_pred, axis=1))
y_true_labels = le.inverse_transform(np.argmax(y_test, axis=1))

print("\n=== Reporte de Clasificación ===")
print(classification_report(y_true_labels, y_pred_labels))

# Guardar predicciones en archivo ARFF
arff_file = "cnn_alzheimer_predictions.arff"
with open(arff_file, 'w') as f:
    f.write("@RELATION cnn_alzheimer_prediction\n\n")
    for i in range(128*128):
        f.write(f"@ATTRIBUTE pixel{i} REAL\n")
    f.write("@ATTRIBUTE class {NonDemented,VeryMildDemented,MildDemented,ModerateDemented}\n\n")
    f.write("@DATA\n")
    for xi, yi in zip(X_test, y_pred_labels):
        flattened = xi.flatten()
        line = ",".join(map(str, flattened)) + f",{yi}\n"
        f.write(line)

print(f"\nPredicciones exportadas a: {arff_file}")
