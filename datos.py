# -*- coding: utf-8 -*-
"""Utilidades de datos para el proyecto microalgas-ia.

Funciones para descargar la colección pública EMDS-7, convertir sus
etiquetas de detección (XML, formato Pascal VOC) en recortes por clase,
preparar particiones train/val/test, aumentar datos y contar células
en fotos de microscopio (línea base con OpenCV).

Uso en Colab:
    import sys
    sys.path.append('/content/microalgas-ia')   # repo clonado
    import datos

Las fotos de Juan viven en su Drive, en:
    /content/drive/MyDrive/microalgas-ia/fotos/<clase>/*.jpg
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# 1. EMDS-7 (Environmental Microorganism Image Dataset, v7)
# ---------------------------------------------------------------------------
# Artículo: Yang et al. 2023, Front. Microbiol. 14:1084312
#   doi:10.3389/fmicb.2023.1084312
# Figshare: https://figshare.com/articles/dataset/EMDS-7_DataSet/16869571
#   DOI del conjunto: 10.6084/m9.figshare.16869571.v1 — Licencia: CC BY 4.0
# Contenido verificado (tabla 1 del artículo): 2,365 imágenes,
# 42 categorías (41 nombradas + "unknown"), 13,216 objetos etiquetados.
# Archivo único: EMDS7.zip, 5,308,086,590 bytes (~5.3 GB).
EMDS7_FIGSHARE_ARTICLE_ID = 16869571
EMDS7_DOWNLOAD_URL = "https://ndownloader.figshare.com/files/31186899"
EMDS7_ZIP_BYTES = 5308086590
EMDS7_PAPER_DOI = "10.3389/fmicb.2023.1084312"


def descargar_emds7(destino_zip, forzar=False):
    """Descarga EMDS7.zip desde Figshare (vía su API pública).

    destino_zip: ruta donde guardar el zip (mejor en Drive, no en el
    disco efímero de Colab: /content/drive/MyDrive/microalgas-ia/EMDS7.zip).
    Devuelve la ruta del zip. Si ya existe y forzar=False, no descarga.
    El archivo pesa ~5.3 GB: la descarga tarda varios minutos.
    """
    import requests
    if os.path.exists(destino_zip) and not forzar:
        print(f"Ya existe {destino_zip}, no se descarga de nuevo.")
        return destino_zip
    api = f"https://api.figshare.com/v2/articles/{EMDS7_FIGSHARE_ARTICLE_ID}"
    info = requests.get(api, timeout=60).json()
    archivos = info.get("files", [])
    if not archivos:
        raise RuntimeError("La API de Figshare no devolvió archivos.")
    url = archivos[0]["download_url"]
    print(f"Descargando {archivos[0]['name']} "
          f"({archivos[0]['size'] / 1e9:.2f} GB) desde Figshare...")
    os.makedirs(os.path.dirname(destino_zip) or ".", exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(destino_zip, "wb") as f:
            for trozo in r.iter_content(chunk_size=1 << 20):
                f.write(trozo)
    print(f"Guardado en {destino_zip}")
    return destino_zip


def descomprimir_zip(ruta_zip, destino):
    """Descomprime un zip (EMDS-7) en `destino`. Salta si ya existe."""
    import zipfile
    if os.path.exists(destino) and os.listdir(destino):
        print(f"Ya existe {destino} con contenido, no se descomprime.")
        return destino
    print(f"Descomprimiendo {ruta_zip} en {destino} ...")
    with zipfile.ZipFile(ruta_zip) as z:
        z.extractall(destino)
    print("Listo.")
    return destino


# ---------------------------------------------------------------------------
# 2. Detección -> clasificación: recortar cada objeto etiquetado
# ---------------------------------------------------------------------------
def _leer_cajas_xml(ruta_xml):
    """Lee un XML Pascal VOC y devuelve [(clase, xmin, ymin, xmax, ymax)]."""
    arbol = ET.parse(ruta_xml)
    raiz = arbol.getroot()
    cajas = []
    for obj in raiz.findall("object"):
        nombre = obj.findtext("name")
        caja = obj.find("bndbox")
        if nombre is None or caja is None:
            continue
        cajas.append((
            nombre.strip(),
            int(float(caja.findtext("xmin"))),
            int(float(caja.findtext("ymin"))),
            int(float(caja.findtext("xmax"))),
            int(float(caja.findtext("ymax"))),
        ))
    return cajas


def extraer_recortes_emds7(carpeta_emds, carpeta_salida,
                           min_objetos=20, excluir=("unknown",),
                           margen=4):
    """Recorta cada objeto etiquetado de EMDS-7 en carpetas por clase.

    carpeta_emds: carpeta con las imágenes y los XML de EMDS-7
        (se buscan recursivamente *.jpg/*.png y *.xml).
    carpeta_salida: se crean subcarpetas <clase>/ con un JPG por objeto.
    min_objetos: clases con menos objetos que esto se descartan
        (con pocos ejemplos el clasificador no aprende nada útil).
    excluir: clases que no son taxones (por defecto "unknown",
        que son impurezas y objetos no identificados).
    margen: píxeles extra alrededor de cada caja.
    Devuelve un dict {clase: nº de recortes}.
    """
    from PIL import Image
    extensiones = (".jpg", ".jpeg", ".png")
    xmls = []
    for raiz, _, archivos in os.walk(carpeta_emds):
        for a in archivos:
            if a.lower().endswith(".xml"):
                xmls.append(os.path.join(raiz, a))
    print(f"XML encontrados: {len(xmls)}")
    conteo = {}
    for ruta_xml in xmls:
        base = os.path.splitext(ruta_xml)[0]
        ruta_img = None
        for ext in extensiones:
            if os.path.exists(base + ext):
                ruta_img = base + ext
                break
        if ruta_img is None:
            continue
        try:
            imagen = Image.open(ruta_img).convert("RGB")
        except Exception:
            continue
        ancho, alto = imagen.size
        for clase, x0, y0, x1, y1 in _leer_cajas_xml(ruta_xml):
            if clase in excluir:
                continue
            x0 = max(0, x0 - margen); y0 = max(0, y0 - margen)
            x1 = min(ancho, x1 + margen); y1 = min(alto, y1 + margen)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            carpeta_clase = os.path.join(carpeta_salida, clase)
            os.makedirs(carpeta_clase, exist_ok=True)
            n = conteo.get(clase, 0)
            recorte = imagen.crop((x0, y0, x1, y1))
            recorte.save(os.path.join(
                carpeta_clase,
                f"{os.path.basename(base)}_{n:04d}.jpg"))
            conteo[clase] = n + 1
    # Descartar clases con muy pocos objetos.
    for clase in [c for c, n in conteo.items() if n < min_objetos]:
        shutil.rmtree(os.path.join(carpeta_salida, clase), ignore_errors=True)
        print(f"Clase descartada por pocos objetos (<{min_objetos}): "
              f"{clase} ({conteo[clase]})")
        del conteo[clase]
    print(f"Clases finales: {len(conteo)} — "
          f"recortes totales: {sum(conteo.values())}")
    return dict(sorted(conteo.items()))


# ---------------------------------------------------------------------------
# 3. Particiones train / val / test
# ---------------------------------------------------------------------------
def preparar_splits(carpeta_recortes, carpeta_splits,
                    test=0.15, val=0.15, semilla=42, max_por_clase=400):
    """Copia los recortes en train/val/test, estratificado por clase.

    max_por_clase: tope de imágenes por clase (submuestreo para que el
    entrenamiento quepa en el nivel gratuito de Colab).
    Devuelve un dict {clase: {'train': n, 'val': n, 'test': n}}.
    """
    rng = random.Random(semilla)
    resumen = {}
    for clase in sorted(os.listdir(carpeta_recortes)):
        origen = os.path.join(carpeta_recortes, clase)
        if not os.path.isdir(origen):
            continue
        archivos = [a for a in os.listdir(origen)
                    if a.lower().endswith((".jpg", ".jpeg", ".png"))]
        rng.shuffle(archivos)
        archivos = archivos[:max_por_clase]
        n = len(archivos)
        n_test = max(1, int(round(n * test)))
        n_val = max(1, int(round(n * val)))
        partes = {
            "test": archivos[:n_test],
            "val": archivos[n_test:n_test + n_val],
            "train": archivos[n_test + n_val:],
        }
        resumen[clase] = {}
        for parte, lista in partes.items():
            destino = os.path.join(carpeta_splits, parte, clase)
            os.makedirs(destino, exist_ok=True)
            for a in lista:
                shutil.copy2(os.path.join(origen, a),
                             os.path.join(destino, a))
            resumen[clase][parte] = len(lista)
    total = {p: sum(v[p] for v in resumen.values()) for p in
             ("train", "val", "test")}
    print(f"Particiones: train={total['train']}, val={total['val']}, "
          f"test={total['test']} — clases: {len(resumen)}")
    return resumen


# ---------------------------------------------------------------------------
# 4. Aumento de datos (data augmentation)
# ---------------------------------------------------------------------------
def capa_aumento():
    """Capas de aumento de datos de Keras.

    De UNA foto se generan MUCHAS variantes: volteada, rotada, con zoom
    y con otro contraste. El modelo aprende a reconocer la célula la vea
    como la vea, y no se memoriza las fotos de entrenamiento.
    Solo se aplica al entrenamiento, nunca a validación ni a test.
    """
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ], name="aumento_datos")


def cargar_datasets(carpeta_splits, tam_img=224, lote=32, semilla=42):
    """Carga train/val/test como tf.data.Dataset desde las carpetas.

    Espera la estructura carpeta_splits/{train,val,test}/<clase>/*.jpg
    Devuelve (train_ds, val_ds, test_ds, nombres_clases).
    """
    import tensorflow as tf
    def _cargar(parte):
        return tf.keras.utils.image_dataset_from_directory(
            os.path.join(carpeta_splits, parte),
            image_size=(tam_img, tam_img),
            batch_size=lote,
            label_mode="int",
            shuffle=(parte == "train"),
            seed=semilla)
    train_ds = _cargar("train")
    val_ds = _cargar("val")
    test_ds = _cargar("test")
    clases = train_ds.class_names
    # Rendimiento: caché + prefetch (recomendado por TensorFlow).
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    print(f"Clases ({len(clases)}): {clases}")
    return train_ds, val_ds, test_ds, clases


# ---------------------------------------------------------------------------
# 5. Conteo de células en fotos de microscopio (línea base OpenCV)
# ---------------------------------------------------------------------------
def contar_celulas_opencv(ruta_imagen, area_min=20, area_max=5000,
                          bloque=51, constante=8, celulas_oscuras=True):
    """Cuenta células en una foto de microscopio (umbral + contornos).

    Método clásico, sin red neuronal: pasa a gris, suaviza, umbral
    adaptativo, busca contornos y filtra por área. Devuelve
    (conteo, imagen_anotada_BGR, areas).
    celulas_oscuras: True (por defecto) si las células se ven OSCURAS
    sobre fondo claro (microscopía de campo claro, lo habitual en
    cultivos); False si se ven CLARAS sobre fondo oscuro (fluorescencia).
    IMPORTANTE: el conteo solo se convierte en densidad (células/mL)
    si se conoce el volumen fotografiado (p. ej. cámara de Neubauer).
    Ver densidad_celular().
    """
    import cv2
    import numpy as np
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer {ruta_imagen}")
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    tipo = (cv2.THRESH_BINARY_INV if celulas_oscuras
            else cv2.THRESH_BINARY)
    binaria = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        tipo, bloque, constante)
    contornos, _ = cv2.findContours(
        binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    validos = []
    for ct in contornos:
        area = cv2.contourArea(ct)
        if area_min <= area <= area_max:
            validos.append(ct)
    anotada = img.copy()
    for ct in validos:
        (x, y), r = cv2.minEnclosingCircle(ct)
        cv2.circle(anotada, (int(x), int(y)), max(2, int(r)), (0, 255, 0), 2)
    areas = [float(cv2.contourArea(ct)) for ct in validos]
    print(f"Células contadas: {len(validos)} "
          f"(área media {np.mean(areas):.1f} px²)" if areas else
          "Células contadas: 0")
    return len(validos), anotada, areas


def densidad_celular(conteo, volumen_ml):
    """Convierte un conteo en densidad: células/mL.

    volumen_ml: volumen de muestra que corresponde a la foto
    (p. ej. un cuadro grande de la cámara de Neubauer = 0.0001 mL).
    Ejemplo: 120 células en 0.0001 mL -> 1,200,000 células/mL.
    """
    if volumen_ml <= 0:
        raise ValueError("El volumen debe ser mayor que cero.")
    return conteo / volumen_ml
