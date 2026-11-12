# Identificación de microalgas y organismos acuáticos con TensorFlow

Proyecto para **identificar microalgas y otros organismos acuáticos en fotos
de microscopio** y **contar células** para estimar la densidad de un cultivo.

## Cómo se usa

**Qué hace:** dada una foto de microscopio, el modelo dice qué organismo es;
y dada una foto de un cultivo, cuenta las células para estimar la densidad
y el estado del cultivo.

**El flujo, en orden:**

1. **Las fotos propias** van en Google Drive, en `microalgas-ia/fotos/<clase>/`
   (una subcarpeta por clase; por ejemplo: `Chlorella_vulgaris`, `Scenedesmus_acutus`).
   Cada foto se etiqueta con lo que se sabe de ella: especie, día, luz, densidad contada
   en Neubauer, estado del cultivo.
2. **Abre `microalgas-ia.ipynb` en Google Colab** (Entorno de ejecución →
   Cambiar tipo de entorno → **GPU**).
3. **Corre las celdas de arriba a abajo.** El cuaderno descarga solo la
   colección pública EMDS-7 (~5.3 GB, la primera vez tarda varios minutos),
   recorta los objetos por clase, suma las fotos propias, entrena el modelo y lo
   evalúa.
4. **El modelo entrenado queda** en `Drive/microalgas-ia/modelos/`:
   `modelo_microalgas.h5` (Keras) y `modelo_microalgas.tflite` (para el
   teléfono, sin internet).

Todo el entrenamiento se ejecuta en Google Colab, con su GPU gratuita. El cuaderno se abre en el navegador, desde cualquier computadora.

**Qué se necesita para empezar:** fotos de microscopio etiquetadas. Sin
fotos propias el cuaderno entrena igual, solo con EMDS-7; las clases propias entran
al entrenamiento en cuanto se suban las fotos y se vuelva a correr.

**Qué descarga el cuaderno por sí solo:** EMDS-7 (colección pública,
2,365 imágenes). WHOI-Plankton no se descarga solo: se baja a mano
(ver abajo).

## Las dos tareas

1. **Identificación:** dada una foto de microscopio, ¿qué organismo es?
   (clasificación por imagen, con transfer learning: EfficientNetB0
   preentrenado en ImageNet + una cabeza nueva para nuestras clases).
2. **Densidad y estado del cultivo:** dada una foto de microscopio, ¿cuántas
   células hay? (línea base con OpenCV: umbral + contornos). El conteo se
   convierte en **células/mL solo si se conoce el volumen fotografiado**
   (por ejemplo, un cuadro de la cámara de Neubauer = 0.0001 mL).

## Archivos

| Archivo | Qué es |
|---|---|
| `microalgas-ia.ipynb` | Cuaderno de Colab (27 celdas, en español): descarga, recortes, splits, aumento, modelo, evaluación, exportación y conteo. |
| `datos.py` | Módulo con el código de datos (descarga EMDS-7, recortes desde XML, splits, aumento, conteo OpenCV). El cuaderno lo importa. |
| `indice-clases.csv` | Índice unificado de clases: 202 filas (EMDS-7, WHOI-Plankton y las fotos propias) con fuente, URL, nº de imágenes/objetos, licencia y notas. |
| `CLASE-paso-a-paso.md` | Clase paso a paso: cómo se construye este modelo desde cero, y cómo adaptar la misma receta a otros modelos (otros organismos, otras tareas). |
| `build_notebook.py` | Generador del cuaderno (el `.ipynb` se regenera con `python3 build_notebook.py`). |
| `smoke_test.py` | Prueba local de todo el flujo con datos sintéticos (sin Colab). |

## Colecciones públicas usadas

### EMDS-7 — la principal (descarga automática en el cuaderno)
- **Qué es:** Environmental Microorganism Image Dataset v7. Yang et al.
  2023, *Front. Microbiol.* 14:1084312, doi:10.3389/fmicb.2023.1084312.
- **Contenido (Tabla 1 del artículo):** **2,365 imágenes**,
  **42 categorías** (41 nombradas + "unknown"), **13,216 objetos
  etiquetados** con cajas (XML, formato Pascal VOC). Incluye *Chlorella*
  (80 imágenes, 155 objetos) y *Scenedesmus* (86 imágenes, 139 objetos).
  La cifra correcta es 2,365 imágenes (no 2,655): la suma de la Tabla 1
  del artículo da exactamente 2,365 imágenes y 13,216 objetos.
- **Descarga:** Figshare, https://figshare.com/articles/dataset/EMDS-7_DataSet/16869571
  — archivo único `EMDS7.zip`, 5,308,086,590 bytes (~5.3 GB). El cuaderno lo
  baja con la API pública de Figshare.
- **Licencia:** CC BY 4.0.

### WHOI-Plankton — referencia (descarga manual, no automática)
- **Qué es:** conjunto anotado de plancton del Woods Hole Oceanographic
  Institution. Orenstein et al. 2015, arXiv:1510.00745. >3.5 millones de
  imágenes en 103 categorías (IFCB, Martha's Vineyard).
- **Enlaces:** https://github.com/hsosik/WHOI-Plankton ·
  DOI 10.1575/1912/7341 · visor de clases https://whoigit.github.io/whoi-plankton
- **Por qué no se descarga en el cuaderno:** la descarga directa está
  **bloqueada por un captcha**: el archivo y los enlaces por año 2006–2014
  redirigen a una página de captcha. Se descarga a mano y se sube el zip a
  Drive. Sus 158 clases del visor oficial están en `indice-clases.csv`.
- **Licencia:** MIT (repositorio); ver DOI 10.1575/1912/7341.

### Plankton Set 1.0 — referencia (descarga manual)
- National Data Science Bowl 2015, NCEI accesión 0127422,
  DOI 10.7289/v5d21vjd, 121 clases. Su enlace de descarga de NCEI es un
  disparador web (no un archivo directo), así que tampoco se descarga en el
  cuaderno. Metadatos:
  https://www.ncei.noaa.gov/access/metadata/landing-page/bin/gfx?id=gov.noaa.nodc:0127422

## Colecciones evaluadas y no incluidas (con el motivo)

- **ADIAC (diatomeas):** ~2,300–3,400 imágenes en ~750 taxones: demasiado
  dispersas por clase para entrenar. Página: https://websites.rbge.org.uk/ADIAC/pubdat/pubdat.html
- **DiatomNet (44 géneros de diatomeas, 2022):** se menciona en un preprint,
  pero no hay una ruta de descarga pública documentada en el proyecto.
  No se incluye.
- **Kaggle (National Data Science Bowl 2015):** requiere cuenta de Kaggle y
  no hay una URL pública del conjunto en el proyecto. No se incluye como ruta
  principal; si se consigue, sus imágenes van a `recortes/<clase>/` igual
  que las fotos propias.

## Límites

- **La identificación de especies necesita fotos de microscopio.** Una foto
  de un cuerpo de agua (río, laguna, dron) no muestra células: ahí solo se
  puede detectar una floración (bloom) o el grupo dominante por el color,
  no la especie.
- **La densidad en células/mL necesita calibración.** El conteo solo se
  convierte en densidad si se conoce el volumen fotografiado (Neubauer).
  Sin el volumen, el número no significa nada: anota siempre con qué
  aumento y qué cámara se tomó la foto.
- **WHOI-Plankton se descarga a mano.** Su descarga directa está bloqueada
  por un captcha; el cuaderno no la baja solo.
- **El modelo solo reconoce lo que ha visto entrenando.** Una clase con 10
  fotos no se aprende: por eso se descartan las clases con menos de 20
  objetos (y la clase "unknown" de EMDS-7, que son impurezas).

## Licencias

EMDS-7: CC BY 4.0 (atribuir a Yang et al. 2023). WHOI-Plankton: MIT (repo).
