# Identificación de microalgas y organismos acuáticos con TensorFlow

Proyecto para **identificar microalgas y otros organismos acuáticos en fotos
de microscopio** y **contar células** para estimar la densidad de un cultivo.
El entrenamiento corre en **Google Colab (nivel gratuito, GPU T4)**; tu
computadora no entrena nada.

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
| `indice-clases.csv` | Índice unificado de clases: 202 filas (EMDS-7, WHOI-Plankton y tus fotos) con fuente, URL, nº de imágenes/objetos, licencia y notas. |
| `build_notebook.py` | Generador del cuaderno (el `.ipynb` se regenera con `python3 build_notebook.py`). |
| `smoke_test.py` | Prueba local de todo el flujo con datos sintéticos (sin Colab). |

## Cómo usarlo (desde cero)

1. **Crea el repositorio en GitHub** (nombre propuesto: `ozono7/microalgas-ia`)
   y sube estos archivos.
2. **En tu Drive**, crea la carpeta `microalgas-ia/` con esta estructura
   (el cuaderno crea lo que falta):

```
Drive/microalgas-ia/
├── fotos/
│   ├── Chlorella_vulgaris/      # tus fotos de microscopio
│   └── Scenedesmus_acutus/
├── recortes/     # lo genera el cuaderno
├── splits/       # train / val / test
└── modelos/      # modelo .h5 y .tflite
```

3. **Sube tus fotos** a `fotos/<clase>/` (una subcarpeta por clase). Etiqueta
   cada foto con lo que sabes: especie, día, luz, densidad contada en
   Neubauer, estado del cultivo.
4. **Abre el cuaderno en Colab**: Entorno de ejecución → Cambiar tipo de
   entorno → **GPU**. Corre las celdas de arriba a abajo. La primera vez
   descarga EMDS-7 (~5.3 GB, tarda varios minutos); las siguientes veces lo
   salta si el zip ya existe.
5. Al final, el modelo queda en `Drive/microalgas-ia/modelos/`:
   `modelo_microalgas.h5` (Keras) y `modelo_microalgas.tflite` (para el
   teléfono, sin internet).

## Colecciones públicas usadas

### EMDS-7 — la principal (descarga automática en el cuaderno)
- **Qué es:** Environmental Microorganism Image Dataset v7. Yang et al.
  2023, *Front. Microbiol.* 14:1084312, doi:10.3389/fmicb.2023.1084312.
- **Contenido verificado (Tabla 1 del artículo):** **2,365 imágenes**,
  **42 categorías** (41 nombradas + "unknown"), **13,216 objetos
  etiquetados** con cajas (XML, formato Pascal VOC). Incluye *Chlorella*
  (80 imágenes, 155 objetos) y *Scenedesmus* (86 imágenes, 139 objetos).
- **Descarga:** Figshare, https://figshare.com/articles/dataset/EMDS-7_DataSet/16869571
  — archivo único `EMDS7.zip`, 5,308,086,590 bytes (~5.3 GB). El cuaderno lo
  baja con la API pública de Figshare (verificado: el enlace responde 206
  y la API devuelve el archivo con ese tamaño exacto).
- **Licencia:** CC BY 4.0.
- **Nota:** el artículo dice 2,365 imágenes (la cifra "2,655" que circula en
  algunos resúmenes es un error: la suma de la Tabla 1 da exactamente 2,365
  imágenes y 13,216 objetos).

### WHOI-Plankton — referencia (descarga MANUAL, no automática)
- **Qué es:** conjunto anotado de plancton del Woods Hole Oceanographic
  Institution. Orenstein et al. 2015, arXiv:1510.00745. >3.5 millones de
  imágenes en 103 categorías (IFCB, Martha's Vineyard).
- **Enlaces verificados:** https://github.com/hsosik/WHOI-Plankton ·
  DOI 10.1575/1912/7341 · visor de clases https://whoigit.github.io/whoi-plankton
- **Por qué no se descarga en el cuaderno:** la descarga directa está
  **bloqueada por un captcha** (verificado 2026-11-12: el archivo y los
  enlaces por año 2006–2014 redirigen a una página de captcha). Se descarga
  a mano y se sube el zip a Drive. Sus 158 clases del visor oficial están en
  `indice-clases.csv`.
- **Licencia:** MIT (repositorio); ver DOI 10.1575/1912/7341.

### Plankton Set 1.0 — referencia (descarga manual)
- National Data Science Bowl 2015, NCEI accesión 0127422,
  DOI 10.7289/v5d21vjd, 121 clases. Su enlace de descarga de NCEI es un
  disparador web (no un archivo directo) y su FTP no devolvió listado, así
  que tampoco se descarga en el cuaderno. Metadatos:
  https://www.ncei.noaa.gov/access/metadata/landing-page/bin/gfx?id=gov.noaa.nodc:0127422

## Evaluadas y NO incluidas (con el motivo)

- **ADIAC (diatomeas):** ~2,300–3,400 imágenes en ~750 taxones: demasiado
  dispersas por clase para entrenar, y no se verificó una URL de descarga
  masiva. Página: https://websites.rbge.org.uk/ADIAC/pubdat/pubdat.html
- **DiatomNet (44 géneros de diatomeas, 2022):** se menciona en un preprint,
  pero su enlace de descarga no se pudo verificar. No se incluye hasta
  verificarlo.
- **Kaggle (National Data Science Bowl 2015):** requiere cuenta de Kaggle y
  no se verificó una URL pública del conjunto. No se incluye como ruta
  principal; si lo consigues, sus imágenes van a `recortes/<clase>/` igual
  que tus fotos.

## Límites honestos

- **La identificación de especies necesita fotos de microscopio.** Una foto
  de un cuerpo de agua (río, laguna, dron) no muestra células: ahí solo se
  puede detectar una floración (bloom) o el grupo dominante por el color,
  no la especie.
- **La densidad en células/mL necesita calibración.** El conteo solo se
  convierte en densidad si se conoce el volumen fotografiado (Neubauer).
  Sin el volumen, el número no significa nada: anota siempre con qué
  aumento y qué cámara se tomó la foto.
- **El modelo solo reconoce lo que ha visto entrenando.** Una clase con 10
  fotos no se aprende: por eso se descartan las clases con menos de 20
  objetos (y la clase "unknown" de EMDS-7, que son impurezas).
- **El aumento de datos para detección no debe recortar** la imagen (cambia
  el número de células). Rotar y voltear sí valen.

## Verificación hecha en este equipo (2026-11-12)

- Cuaderno: JSON válido, 27 celdas (13 markdown, 14 código); todas las
  celdas de código compilan; sin comandos de shell (`!`/`%`) para que el
  código sea revisable.
- Smoke test con TensorFlow 2.21.0 (CPU local): recortes desde XML, splits
  70/15/15, EfficientNetB0 (1 época sobre datos sintéticos), evaluación,
  export `.h5` (16,335 KB) y `.tflite` (15,695 KB), conteo OpenCV
  **100 de 100 células sintéticas** (densidad 1,000,000 cél/mL con
  Neubauer 0.0001 mL), y metadatos EMDS-7 vía API de Figshare.
- **No se entrenó el modelo real aquí** (eso se hace en Colab, con GPU y
  con EMDS-7 descargado): la primera corrida en Colab es la prueba de
  entrenamiento real.
- URLs verificadas con peticiones de prueba (sin descargar gigabytes):
  EMDS-7 (206, `application/zip`), API de Figshare, GitHub
  `hsosik/WHOI-Plankton`, visor WHOI, metadatos NCEI.

## Licencias

EMDS-7: CC BY 4.0 (atribuir a Yang et al. 2023). WHOI-Plankton: MIT (repo).
Código de este proyecto: por definir (se sugiere MIT al crear el repo).
