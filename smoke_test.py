# -*- coding: utf-8 -*-
"""Smoke test local del proyecto microalgas-ia (sin Colab, sin descargas grandes).

- Extracción de recortes EMDS-7 sobre un mini-conjunto falso.
- Splits + carga de datasets + modelo EfficientNetB0 + 1 época + evaluación.
- Export .h5 y .tflite.
- Conteo OpenCV sobre imagen sintética.
- Metadatos de EMDS-7 vía API de Figshare (sin descargar el zip).
"""
import os
import sys
import shutil

sys.path.insert(0, "/home/hatch/workspace/microalgas-ia")
import datos

BASE = "/tmp/smoke-microalgas"
shutil.rmtree(BASE, ignore_errors=True)
os.makedirs(BASE, exist_ok=True)

# --- 1. Mini EMDS-7 falso: 1 imagen + XML con 3 cajas ----------------------
from PIL import Image, ImageDraw
import numpy as np

img = Image.new("RGB", (200, 200), (40, 40, 40))
d = ImageDraw.Draw(img)
d.ellipse([20, 20, 80, 80], fill=(220, 220, 220))   # Chlorella
d.ellipse([100, 100, 170, 150], fill=(200, 200, 200))  # Scenedesmus
img.save(os.path.join(BASE, "foto1.jpg"))
xml = """<annotation><filename>foto1.jpg</filename><size><width>200</width><height>200</height></size>
<object><name>Chlorella</name><bndbox><xmin>20</xmin><ymin>20</ymin><xmax>80</xmax><ymax>80</ymax></bndbox></object>
<object><name>Scenedesmus</name><bndbox><xmin>100</xmin><ymin>100</ymin><xmax>170</xmax><ymax>150</ymax></bndbox></object>
<object><name>unknown</name><bndbox><xmin>5</xmin><ymin>150</ymin><xmax>40</xmax><ymax>180</ymax></bndbox></object>
</annotation>"""
open(os.path.join(BASE, "foto1.xml"), "w").write(xml)
conteo = datos.extraer_recortes_emds7(BASE, os.path.join(BASE, "recortes"),
                                       min_objetos=1, excluir=("unknown",))
assert conteo == {"Chlorella": 1, "Scenedesmus": 1}, conteo
print("OK recortes:", conteo)

# --- 2. Dataset sintético de 3 clases para entrenar -------------------------
rng = np.random.default_rng(3)
CLASES_S = ["Chlorella", "Scenedesmus", "Microcystis"]
for cl in CLASES_S:
    os.makedirs(os.path.join(BASE, "recortes", cl), exist_ok=True)
    for i in range(12):
        arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(BASE, "recortes", cl, f"img{i:02d}.jpg"))
resumen = datos.preparar_splits(os.path.join(BASE, "recortes"),
                                 os.path.join(BASE, "splits"),
                                 test=0.15, val=0.15, semilla=42,
                                 max_por_clase=400)
print("OK splits:", resumen)

import tensorflow as tf
train_ds, val_ds, test_ds, CLASES = datos.cargar_datasets(
    os.path.join(BASE, "splits"), tam_img=224, lote=8, semilla=42)
print("OK datasets, clases:", CLASES)

# --- 3. Modelo (mismo código que el cuaderno) -------------------------------
aumento = datos.capa_aumento()
train_ds_aug = train_ds.map(lambda x, y: (aumento(x, training=True), y))
NUM_CLASES = len(CLASES)
base = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base.trainable = False
entradas = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.efficientnet.preprocess_input(entradas)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
salidas = tf.keras.layers.Dense(NUM_CLASES, activation="softmax")(x)
modelo = tf.keras.Model(entradas, salidas)
modelo.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])
print("OK modelo construido")
historial = modelo.fit(train_ds_aug, validation_data=val_ds, epochs=1, verbose=0)
print("OK 1 época entrenada — loss:", round(historial.history["loss"][0], 4))
perdida, exactitud = modelo.evaluate(test_ds, verbose=0)
print(f"OK evaluación test — pérdida: {perdida:.4f} exactitud: {exactitud:.4f}")

# --- 4. Export .h5 y .tflite -------------------------------------------------
ruta_h5 = os.path.join(BASE, "modelo_microalgas.h5")
modelo.save(ruta_h5)
print("OK .h5:", os.path.getsize(ruta_h5) // 1024, "KB")
convertidor = tf.lite.TFLiteConverter.from_keras_model(modelo)
tflite = convertidor.convert()
ruta_tflite = os.path.join(BASE, "modelo_microalgas.tflite")
open(ruta_tflite, "wb").write(tflite)
print("OK .tflite:", os.path.getsize(ruta_tflite) // 1024, "KB")

# --- 5. Conteo OpenCV sobre imagen sintética -------------------------------
import cv2
demo = np.zeros((600, 800, 3), dtype=np.uint8) + 220
rng2 = np.random.default_rng(7)
for fila in range(10):
    for col in range(10):
        cx = col * 80 + 40 + int(rng2.integers(-10, 11))
        cy = fila * 60 + 30 + int(rng2.integers(-10, 11))
        r = int(rng2.integers(6, 13))
        cv2.circle(demo, (cx, cy), r, (40, 40, 40), -1)
cv2.imwrite(os.path.join(BASE, "demo.jpg"), demo)
conteo_c, anotada, areas = datos.contar_celulas_opencv(
    os.path.join(BASE, "demo.jpg"), area_min=100, area_max=800)
assert conteo_c >= 98, f"conteo demasiado bajo: {conteo_c} de 100"
print(f"OK conteo: {conteo_c} de 100 sintéticas — "
      f"densidad Neubauer: {datos.densidad_celular(conteo_c, 0.0001):,.0f} cél/mL")

# --- 6. Metadatos EMDS-7 vía API de Figshare (sin descargar) ----------------
import requests
info = requests.get(
    f"https://api.figshare.com/v2/articles/{datos.EMDS7_FIGSHARE_ARTICLE_ID}",
    timeout=60).json()
f0 = info["files"][0]
print("OK Figshare API — archivo:", f0["name"], "| bytes:", f0["size"],
      "| coincide con lo documentado:", f0["size"] == datos.EMDS7_ZIP_BYTES)
print("SMOKE TEST COMPLETO")
