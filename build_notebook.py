# -*- coding: utf-8 -*-
"""Genera microalgas-ia.ipynb (cuaderno de Colab, JSON válido)."""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "microalgas-ia.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

cells.append(md('''# Identificación de microalgas y organismos acuáticos con TensorFlow

Cuaderno para entrenar un modelo que **identifique microalgas y otros organismos
acuáticos en fotos de microscopio**, y para **contar células** y estimar la
densidad de un cultivo. Corre en **Google Colab (nivel gratuito, GPU T4)**.

**Las dos tareas del proyecto**

1. **Identificación:** dada una foto de microscopio, ¿qué organismo es?
   (clasificación por imagen).
2. **Densidad y estado del cultivo:** dada una foto de microscopio,
   ¿cuántas células hay y cómo está el cultivo? (conteo + densidad en
   células/mL, con calibración).

**De dónde salen los datos**

- **EMDS-7** (colección pública principal): 2,365 imágenes de microscopio,
  41 tipos de microorganismos + "unknown", 13,216 objetos etiquetados.
  Licencia CC BY 4.0. Incluye *Chlorella* y *Scenedesmus*.
- **WHOI-Plankton** (referencia): >3.5 millones de imágenes, 103 categorías.
  Su descarga directa está bloqueada por un captcha, así que
  en este cuaderno **no** se descarga: se documenta cómo obtenerla a mano.
- **Fotos propias:** fotos de microscopio de cultivos (por ejemplo,
  *Chlorella vulgaris* y *Scenedesmus acutus*), en Google Drive. Son las clases propias del proyecto.

**Límites honestos (léelos antes de entrenar)**

- La **identificación de especies necesita fotos de microscopio**. Una foto
  de un cuerpo de agua (río, laguna, dron) no muestra células: ahí solo se
  puede detectar una floración (bloom) o el grupo dominante por el color,
  no la especie.
- La **densidad en células/mL necesita calibración**: el modelo cuenta
  células en la foto, y ese conteo solo se convierte en densidad si se
  conoce el volumen fotografiado (por ejemplo, un cuadro de la cámara de
  Neubauer = 0.0001 mL).
- El modelo solo reconoce lo que ha visto entrenando. Una clase con 10 fotos
  no se aprende: por eso se descartan las clases con menos de 20 objetos.
'''))

cells.append(md('''## Antes de empezar: qué necesitas

1. Una cuenta de Google. Abre este cuaderno en **Colab** (Entorno de
   ejecución → Cambiar tipo de entorno → **GPU**). Todo el entrenamiento
   se ejecuta en Colab, con su GPU gratuita. El cuaderno se abre en el
   navegador, desde cualquier computadora.
2. Las fotos propias en Google Drive, en `microalgas-ia/fotos/<clase>/` (una
   subcarpeta por clase):

```
Drive/microalgas-ia/
├── fotos/                  # fotos propias de microscopio
│   ├── Chlorella_vulgaris/
│   └── Scenedesmus_acutus/
├── recortes/               # lo genera el cuaderno (EMDS-7 por clase)
├── splits/                 # train / val / test (lo genera el cuaderno)
└── modelos/                # modelo .h5 y .tflite (lo genera el cuaderno)
```

3. El repositorio `ozono7/microalgas-ia` en GitHub (ahí viven este cuaderno,
   `datos.py`, el `README.md` y el `indice-clases.csv`). El cuaderno lo clona
   solo.
'''))

cells.append(code('''# 0. Comprobar GPU y versiones
import subprocess
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("GPU disponible:", tf.config.list_physical_devices("GPU"))
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                          "--format=csv"], capture_output=True,
                         text=True, timeout=30).stdout)
except Exception as e:
    print("nvidia-smi no disponible:", e)
'''.strip()))

cells.append(code('''# 1. Montar Drive y clonar el repo
import os
import sys
import subprocess

from google.colab import drive
drive.mount("/content/drive")

REPO_DIR = "/content/microalgas-ia"
if not os.path.exists(os.path.join(REPO_DIR, "datos.py")):
    r = subprocess.run(["git", "clone",
                        "https://github.com/ozono7/microalgas-ia.git",
                        REPO_DIR], capture_output=True, text=True)
    print(r.stdout[-500:] if r.stdout else "")
    print(r.stderr[-500:] if r.stderr else "")
if not os.path.exists(os.path.join(REPO_DIR, "datos.py")):
    raise FileNotFoundError(
        "No se encontró datos.py en /content/microalgas-ia.")
sys.path.insert(0, REPO_DIR)

import datos
print("Módulo datos OK — EMDS-7:", datos.EMDS7_FIGSHARE_ARTICLE_ID)
'''.strip()))

cells.append(md('''## 2. Descargar EMDS-7 (colección pública principal)

- **Qué es:** Environmental Microorganism Image Dataset v7 (Yang et al.
  2023, *Front. Microbiol.* 14:1084312, doi:10.3389/fmicb.2023.1084312).
- **Contenido:** 2,365 imágenes, 42 categorías (41 nombradas +
  "unknown"), 13,216 objetos etiquetados con cajas (XML, formato Pascal VOC).
- **Licencia:** CC BY 4.0 (uso no comercial con atribución).
- **Tamaño:** el zip pesa ~5.3 GB. Se descarga al disco **efímero** de Colab
  (`/content`, rápido) y solo los recortes pequeños se guardan en Drive.
  Si la sesión termina, se vuelve a descargar (el código lo salta si el zip
  ya existe).
'''))

cells.append(code('''# 2. Descargar y descomprimir EMDS-7 (tarda varios minutos, ~5.3 GB)
BASE_DRIVE = "/content/drive/MyDrive/microalgas-ia"
ZIP_LOCAL = "/content/EMDS7.zip"
DIR_EMDS = "/content/EMDS7"

datos.descargar_emds7(ZIP_LOCAL)
datos.descomprimir_zip(ZIP_LOCAL, DIR_EMDS)
'''.strip()))

cells.append(md('''## 3. De detección a clasificación: recortar cada objeto

EMDS-7 es un conjunto de **detección**: cada foto tiene varias cajas con su
clase. Para entrenar un **clasificador** recortamos cada caja y guardamos un
JPG por objeto, en carpetas por clase (`recortes/<clase>/`).

- Se excluye la clase `"unknown"` (impurezas y objetos no identificados:
  no es un taxón).
- Se descartan las clases con menos de 20 objetos (con tan pocos ejemplos
  el modelo no aprende nada útil).
'''))

cells.append(code('''# 3. Recortar objetos de EMDS-7 por clase (se guarda en Drive)
import matplotlib.pyplot as plt

RECORTES = os.path.join(BASE_DRIVE, "recortes")
conteo = datos.extraer_recortes_emds7(DIR_EMDS, RECORTES,
                                       min_objetos=20,
                                       excluir=("unknown",))
print(conteo)

# Gráfico: ¿cuántos ejemplos tiene cada clase?
clases = list(conteo.keys())
valores = list(conteo.values())
plt.figure(figsize=(10, 5))
plt.bar(clases, valores)
plt.xticks(rotation=90)
plt.ylabel("Objetos (recortes)")
plt.title("EMDS-7: recortes por clase")
plt.tight_layout()
plt.show()
'''.strip()))

cells.append(md('''## 4. Fotos propias (opcional, pero son las clases propias)

Si ya hay fotos en `Drive/microalgas-ia/fotos/<clase>/`, esta celda las
copia al conjunto de recortes. Cada subcarpeta es una clase, por ejemplo:

- `fotos/Chlorella_vulgaris/` — fotos de *Chlorella vulgaris*
- `fotos/Scenedesmus_acutus/` — fotos de *Scenedesmus acutus*

Cada foto se etiqueta con lo que se sabe de ella (especie, día, luz, densidad contada
en Neubauer, estado del cultivo) en el nombre del archivo o en una tabla
aparte. Si aún no hay fotos, la celda lo dice y seguimos solo con EMDS-7.
'''))

cells.append(code('''# 4. Agregar las fotos propias de Drive al conjunto (si existen)
import shutil

FOTOS = os.path.join(BASE_DRIVE, "fotos")
agregadas = {}
if os.path.isdir(FOTOS):
    for clase in sorted(os.listdir(FOTOS)):
        origen = os.path.join(FOTOS, clase)
        if not os.path.isdir(origen):
            continue
        destino = os.path.join(RECORTES, clase)
        os.makedirs(destino, exist_ok=True)
        n = 0
        for a in os.listdir(origen):
            if a.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.copy2(os.path.join(origen, a),
                             os.path.join(destino, a))
                n += 1
        agregadas[clase] = n
        print(f"{clase}: {n} fotos agregadas")
if not agregadas:
    print("Aún no hay fotos en Drive/microalgas-ia/fotos/. "
          "Se entrena solo con EMDS-7.")
'''.strip()))

cells.append(md('''## 5. WHOI-Plankton (referencia: descarga manual, no automática)

**WHOI-Plankton** (Orenstein et al. 2015, arXiv:1510.00745) es el conjunto de
plancton anotado más grande: >3.5 millones de imágenes en 103 categorías,
tomadas con el Imaging FlowCytobot en Martha's Vineyard.

La descarga directa **no funciona de forma automática**: el archivo
(DOI 10.1575/1912/7341) redirige a una página con captcha, y los enlaces
directos por año también. Por eso este cuaderno **no intenta descargarlo**:
lo documenta para obtenerlo a mano:

1. Entrar a https://github.com/hsosik/WHOI-Plankton (DOI 10.1575/1912/7341).
2. Descargar a mano el año que quieras (2006–2014) y subir el zip a
   `Drive/microalgas-ia/`.
3. Las 158 clases del visor oficial están en `indice-clases.csv`.

También existe **Plankton Set 1.0** (National Data Science Bowl 2015, NCEI
accesión 0127422, DOI 10.7289/v5d21vjd, 121 clases): su enlace de descarga
de NCEI es un disparador web, no un archivo directo; por eso tampoco se
descarga aquí. Si lo consigues (Kaggle, con cuenta), sus imágenes van a
`recortes/<clase>/` igual que las fotos propias.
'''))

cells.append(code('''# 5. Enlaces de WHOI-Plankton (solo informativos)
print("WHOI-Plankton — repositorio y DOI:")
print("  https://github.com/hsosik/WHOI-Plankton")
print("  http://dx.doi.org/10.1575/1912/7341")
print("  Visor de clases: https://whoigit.github.io/whoi-plankton")
print("Plankton Set 1.0 — metadatos NCEI:")
print("  https://www.ncei.noaa.gov/access/metadata/landing-page/bin/gfx?id=gov.noaa.nodc:0127422")
print("Nota: la descarga directa está bloqueada por captcha; bajar a mano.")
'''.strip()))

cells.append(md('''## 6. Particiones train / val / test

Separamos las imágenes por clase en tres grupos, con semilla fija para que
el experimento sea repetible:

- **train (70%)**: con lo que aprende el modelo.
- **val (15%)**: para vigilar el entrenamiento y parar a tiempo.
- **test (15%)**: solo se toca al final, para medir de verdad.

Para que el entrenamiento quepa en el nivel gratuito se limitan las clases
a 400 imágenes como máximo (`max_por_clase=400`).
'''))

cells.append(code('''# 6. Particiones y carga de los conjuntos
SPLITS = os.path.join(BASE_DRIVE, "splits")
resumen = datos.preparar_splits(RECORTES, SPLITS,
                                test=0.15, val=0.15,
                                semilla=42, max_por_clase=400)

train_ds, val_ds, test_ds, CLASES = datos.cargar_datasets(
    SPLITS, tam_img=224, lote=32, semilla=42)
print(f"Número de clases: {len(CLASES)}")
'''.strip()))

cells.append(md('''## 7. Aumento de datos (data augmentation)

**De una foto se hacen muchas variantes.** El aumento aplica transformaciones
aleatorias cada vez que el modelo ve una imagen:

- voltear (horizontal y vertical),
- rotar,
- zoom,
- cambiar el contraste.

Así el modelo aprende a reconocer la célula la vea como la vea, y no se
memoriza las fotos de entrenamiento. El aumento **solo** se aplica al
entrenamiento, nunca a validación ni a test (ahí se mide con fotos reales).

> **Ojo con el conteo:** para clasificar, el aumento vale. Para **detectar
> y contar** células, el aumento **no debe recortar** la imagen, porque
> cambiaría el número de células. Rotar y voltear sí valen (no cambian
> el conteo).
'''))

cells.append(code('''# 7. Aumento de datos: ver una foto y sus variantes
import matplotlib.pyplot as plt

aumento = datos.capa_aumento()
train_ds_aug = train_ds.map(
    lambda x, y: (aumento(x, training=True), y))

# Una imagen original y 4 variantes aumentadas
for imagenes, etiquetas in train_ds.take(1):
    img = imagenes[0]
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 5, 1)
    plt.imshow(img.numpy().astype("uint8"))
    plt.title(f"Original\\n{CLASES[etiquetas[0].numpy()]}")
    plt.axis("off")
    for i in range(4):
        variante = aumento(tf.expand_dims(img, 0), training=True)[0]
        plt.subplot(1, 5, i + 2)
        plt.imshow(variante.numpy().astype("uint8"))
        plt.title(f"Variante {i + 1}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    break
'''.strip()))

cells.append(md('''## 8. Modelo: EfficientNetB0 con transfer learning

No entrenamos desde cero: partimos de **EfficientNetB0** entrenado con
ImageNet (millones de fotos) y solo entrenamos una cabeza clasificadora
nueva para nuestras clases. Es lo estándar cuando hay pocas imágenes por
clase.

- La base queda **congelada** (sus pesos no cambian).
- Encima: promedio global + dropout (0.3, contra el sobreajuste) + capa
  densa con tantas salidas como clases.

Si después quieres más precisión: descongelar las últimas capas de la base
y reentrenar con una tasa de aprendizaje baja (fine-tuning).
'''))

cells.append(code('''# 8. Construir el modelo
import tensorflow as tf

NUM_CLASES = len(CLASES)
base = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet")
base.trainable = False  # base congelada (transfer learning)

entradas = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.efficientnet.preprocess_input(entradas)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
salidas = tf.keras.layers.Dense(NUM_CLASES, activation="softmax")(x)
modelo = tf.keras.Model(entradas, salidas)

modelo.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
modelo.summary()
'''.strip()))

cells.append(code('''# 8b. Entrenar (con parada temprana)
import matplotlib.pyplot as plt

EPOCAS = 10
historial = modelo.fit(
    train_ds_aug,
    validation_data=val_ds,
    epochs=EPOCAS,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        patience=3, restore_best_weights=True, monitor="val_loss")])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(historial.history["accuracy"], label="train")
plt.plot(historial.history["val_accuracy"], label="val")
plt.title("Exactitud"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(historial.history["loss"], label="train")
plt.plot(historial.history["val_loss"], label="val")
plt.title("Pérdida"); plt.legend()
plt.tight_layout()
plt.show()
'''.strip()))

cells.append(md('''## 9. Evaluación

La medida que importa es la del conjunto **test** (fotos que el modelo
nunca vio). Además de la exactitud global, miramos el reporte por clase
(qué clases confunde) y la matriz de confusión.
'''))

cells.append(code('''# 9. Evaluar en test + reporte por clase + matriz de confusión
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

perdida, exactitud = modelo.evaluate(test_ds)
print(f"Test — pérdida: {perdida:.4f} — exactitud: {exactitud:.4f}")

y_real, y_pred = [], []
for imagenes, etiquetas in test_ds:
    probs = modelo.predict(imagenes, verbose=0)
    y_pred.extend(np.argmax(probs, axis=1))
    y_real.extend(etiquetas.numpy())
print(classification_report(y_real, y_pred, target_names=CLASES, zero_division=0))

mat = confusion_matrix(y_real, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(mat, cmap="Blues")
plt.colorbar()
plt.xticks(range(len(CLASES)), CLASES, rotation=90)
plt.yticks(range(len(CLASES)), CLASES)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de confusión (test)")
plt.tight_layout()
plt.show()
'''.strip()))

cells.append(md('''## 10. Exportar el modelo

Se guardan dos archivos en `Drive/microalgas-ia/modelos/`:

- **`modelo_microalgas.h5`**: el modelo completo de Keras, para seguir
  entrenándolo o usarlo en Python.
- **`modelo_microalgas.tflite`**: versión ligera (TensorFlow Lite) para
  correrlo **en el teléfono** en el campo, sin internet.
'''))

cells.append(code('''# 10. Exportar .h5 y .tflite a Drive
import os

MODELOS = os.path.join(BASE_DRIVE, "modelos")
os.makedirs(MODELOS, exist_ok=True)
ruta_h5 = os.path.join(MODELOS, "modelo_microalgas.h5")
modelo.save(ruta_h5)
print("Guardado:", ruta_h5, os.path.getsize(ruta_h5) // 1024, "KB")

convertidor = tf.lite.TFLiteConverter.from_keras_model(modelo)
tflite = convertidor.convert()
ruta_tflite = os.path.join(MODELOS, "modelo_microalgas.tflite")
with open(ruta_tflite, "wb") as f:
    f.write(tflite)
print("Guardado:", ruta_tflite, os.path.getsize(ruta_tflite) // 1024, "KB")

with open(os.path.join(MODELOS, "clases.txt"), "w") as f:
    f.write("\\n".join(CLASES))
print("Clases guardadas en clases.txt (el orden importa para predecir).")
'''.strip()))

cells.append(md('''## 11. Densidad celular desde fotos de microscopio

El conteo no necesita red neuronal para empezar: una **línea base con
OpenCV** (pasar a gris → suavizar → umbral adaptativo → buscar contornos →
filtrar por área) ya cuenta células en fotos limpias de cultivos.

**La calibración es lo que convierte el conteo en densidad:**

```
densidad (células/mL) = células contadas / volumen fotografiado (mL)
```

Ejemplo: 120 células en un cuadro grande de la cámara de Neubauer
(0.0001 mL) = 1,200,000 células/mL. Sin el volumen, el número no significa
nada: anota siempre con qué aumento y qué cámara se tomó la foto.

Abajo se demuestra con una **imagen sintética** (células oscuras sobre
fondo claro, como en campo claro). Después, la misma función corre sobre
las fotos propias de Drive.
'''))

cells.append(code('''# 11. Conteo de células: demo con imagen sintética + una foto propia (si hay)
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Imagen sintética: 100 "células" oscuras en rejilla con jitter (campo claro)
# (rejilla para que no se solapen: el conteo debe salir exacto)
rng = np.random.default_rng(7)
demo = np.zeros((600, 800, 3), dtype=np.uint8) + 220
for fila in range(10):
    for col in range(10):
        cx = col * 80 + 40 + int(rng.integers(-10, 11))
        cy = fila * 60 + 30 + int(rng.integers(-10, 11))
        r = int(rng.integers(6, 13))
        cv2.circle(demo, (cx, cy), r, (40, 40, 40), -1)
cv2.imwrite("/tmp/demo_microscopio.jpg", demo)

conteo, anotada, areas = datos.contar_celulas_opencv(
    "/tmp/demo_microscopio.jpg", area_min=100, area_max=800)
print("Densidad si el campo fuera un cuadro de Neubauer (0.0001 mL):",
      f"{datos.densidad_celular(conteo, 0.0001):,.0f} células/mL")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(demo, cv2.COLOR_BGR2RGB))
plt.title("Sintética (100 células)")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(anotada, cv2.COLOR_BGR2RGB))
plt.title(f"Contadas: {conteo}")
plt.axis("off")
plt.tight_layout()
plt.show()
'''.strip()))

cells.append(code('''# 11b. Conteo sobre una primera foto propia de Drive (si ya hay alguna)
import glob

candidatas = sorted(glob.glob(os.path.join(BASE_DRIVE, "fotos", "*", "*.jpg")) +
                    glob.glob(os.path.join(BASE_DRIVE, "fotos", "*", "*.png")))
if candidatas:
    ruta = candidatas[0]
    print("Foto:", ruta)
    conteo, anotada, areas = datos.contar_celulas_opencv(ruta)
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(anotada, cv2.COLOR_BGR2RGB))
    plt.title(f"{os.path.basename(ruta)} — contadas: {conteo}")
    plt.axis("off")
    plt.show()
    print("Ajusta area_min/area_max y el umbral hasta que el conteo "
          "coincida con el conteo en Neubauer, y anota esos valores.")
else:
    print("Aún no hay fotos en Drive/microalgas-ia/fotos/.")
'''.strip()))

cells.append(md('''## 12. Clasificar una foto nueva

Ya entrenado el modelo, esta es la parte que se usa todos los días: subir
una foto de microscopio y ver qué organismo dice el modelo que es, con su
confianza en %. Funciona de dos formas: subir una foto con el botón, o
escribir la ruta de una foto que ya esté en Drive (por ejemplo, una de
`microalgas-ia/fotos/`). Si el cuaderno se acaba de abrir y el modelo no
está en memoria, se carga solo el archivo exportado en
`Drive/microalgas-ia/modelos/modelo_microalgas.h5`.
'''))

cells.append(code('''# 12. Clasificar una foto nueva: subir una imagen y ver qué organismo es
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from google.colab import files

# Si el modelo no está en memoria (cuaderno recién abierto), se carga el
# exportado en Drive.
try:
    modelo
except NameError:
    ruta_h5 = os.path.join(BASE_DRIVE, "modelos", "modelo_microalgas.h5")
    modelo = tf.keras.models.load_model(ruta_h5)
    with open(os.path.join(BASE_DRIVE, "modelos", "clases.txt")) as f:
        CLASES = [l.strip() for l in f if l.strip()]
    print("Modelo cargado de Drive:", ruta_h5)

# Ruta de una foto en Drive (vacío = subir una con el botón).
# Ejemplo: "Drive/microalgas-ia/fotos/Chlorella_vulgaris/foto1.jpg"
RUTA_FOTO = ""
if not RUTA_FOTO:
    print("Sube una foto de microscopio:")
    subidas = files.upload()
    RUTA_FOTO = list(subidas.keys())[0]

img = Image.open(RUTA_FOTO).convert("RGB").resize((224, 224))
x = np.expand_dims(np.array(img), 0)
probs = modelo.predict(x, verbose=0)[0]
top3 = np.argsort(probs)[::-1][:3]

plt.figure(figsize=(6, 6))
plt.imshow(Image.open(RUTA_FOTO).convert("RGB"))
plt.axis("off")
plt.title("Foto: " + os.path.basename(RUTA_FOTO))
plt.show()

print("El modelo dice:")
for i in top3:
    print(f"  {CLASES[i]}: {probs[i] * 100:.1f}%")
'''.strip()))

cells.append(md('''## Qué sigue

1. **Subir las fotos propias** a `Drive/microalgas-ia/fotos/<clase>/` (unas cientos
   por clase para empezar) y volver a correr desde la celda 4: las clases propias
   entran al entrenamiento automáticamente.
2. **Más épocas / fine-tuning:** descongela las últimas capas de
   EfficientNetB0 y reentrena con tasa baja.
3. **Detección (no solo clasificación):** para contar por clase en una foto
   con varias especies, el paso siguiente es un detector (YOLO) entrenado
   con las cajas XML de EMDS-7, que ya tienes.
4. **WHOI-Plankton:** descarga manual (ver sección 5) para ampliar las
   clases marinas.
5. **En el teléfono:** el `.tflite` exportado corre en Android/iOS sin
   internet, para sondear cuerpos de agua en campo.
'''))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Escrito {OUT} — {len(cells)} celdas")
