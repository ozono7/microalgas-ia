# Clase paso a paso: cómo se construye un modelo que identifica microalgas

Esta clase explica, desde cero, cómo se construye el modelo de este
proyecto — y cómo adaptar la misma receta para formar **otros modelos**:
otros organismos, otras tareas (clasificar estados, contar, detectar).
No es una copia del cuaderno: es la receta, con el qué y el porqué de
cada paso. El cuaderno `microalgas-ia.ipynb` es la receta ya ejecutada;
aquí está el razonamiento.

## Qué vas a aprender

- Qué necesita un modelo para aprender a reconocer algo en una foto.
- Por qué se divide el trabajo en fotos, carpetas, particiones y aumento.
- Por qué no se entrena desde cero (transfer learning).
- Cómo se mide si el modelo sirve (y qué hacer si no sirve).
- Cómo se cuenta la densidad de un cultivo desde una foto.
- Cómo cambiar la receta para otro problema.

## Qué necesitas antes de empezar

- Fotos de microscopio, etiquetadas. Sin fotos no hay modelo: el modelo
  aprende de ejemplos, no de definiciones.
- Una cuenta de Google y Google Colab con GPU (gratuito). Todo el
  entrenamiento se hace en Colab; en tu MacBook solo abres el cuaderno
  en el navegador.
- Este repositorio (`ozono7/microalgas-ia`) y tu carpeta de Drive
  `microalgas-ia/fotos/`.

---

## Paso 1. Juntar las fotos (el paso que decide todo)

**Qué se hace:** tomar fotos de microscopio de cada organismo que el
modelo debe reconocer, y anotar para cada foto lo que tú sabes: especie,
día, luz, densidad contada en Neubauer, estado del cultivo.

**Por qué:** el modelo no sabe biología. Lo único que aprende es la
relación entre "cómo se ve" y "qué es". Si una foto no tiene etiqueta,
no sirve para entrenar. Si la etiqueta está mal, el modelo aprende el
error.

**Reglas prácticas:**

- Una foto por clase no basta: unas cientos por clase para empezar.
- Las fotos deben parecerse a las que el modelo verá después: mismo
  aumento, misma iluminación, mismo tipo de muestra.
- Anota siempre con qué aumento y qué cámara se tomó la foto: sin ese
  dato, el conteo no se puede convertir en densidad.

**En el cuaderno:** sección 4 (tus fotos) y sección 11 (conteo).

## Paso 2. Organizar por carpetas (una carpeta = una clase)

**Qué se hace:** una subcarpeta por clase dentro de
`microalgas-ia/fotos/` (por ejemplo, `Chlorella_vulgaris/`,
`Scenedesmus_acutus/`). Cada foto va en la carpeta de su clase.

**Por qué:** el nombre de la carpeta ES la etiqueta. Así el código no
necesita una tabla aparte: lee la carpeta y ya sabe qué es cada foto.
Es la convención que usan TensorFlow y casi todos los conjuntos de
imágenes.

**En el cuaderno:** secciones 3 (recortes por clase) y 4.

## Paso 3. Dividir en tres grupos: train, val, test

**Qué se hace:** las fotos de cada clase se reparten en tres grupos:
train (70%), val (15%) y test (15%).

**Por qué:** si el modelo se evalúa con las mismas fotos con las que
aprendió, la nota sale inflada: se las memorizó. Por eso se separan:

- **train:** con lo que aprende el modelo.
- **val:** para vigilar el entrenamiento y parar a tiempo (si deja de
  mejorar, se para: eso es la "parada temprana").
- **test:** solo se toca al final, para medir de verdad con fotos que
  el modelo nunca vio.

**Regla de oro:** una foto no puede estar en dos grupos a la vez. Y si
tienes varias fotos del mismo cultivo o de la misma sesión, repártelas
con cuidado: si todas las fotos de un cultivo caen en train y ninguna
en test, la medida final no dice nada de ese cultivo.

**En el cuaderno:** sección 6.

## Paso 4. Aumento de datos (una foto cuenta por varias)

**Qué se hace:** cada vez que el modelo ve una foto durante el
entrenamiento, la ve transformada: volteada, rotada, con zoom o con
otro contraste. De una foto salen muchas variantes.

**Por qué:** con pocas fotos, el modelo se memoriza las fotos exactas
en vez de aprender qué hace que una célula sea de una especie. Las
variantes lo obligan a aprender la forma, no la foto concreta. Es la
técnica que permite entrenar con cientos de fotos en vez de miles.

**Qué transformaciones valen y cuáles no:**

- Voltear, rotar, zoom y contraste: valen para clasificar.
- **Para contar células, el aumento no debe recortar** la imagen: un
  recorte cambia el número de células de la foto. Rotar y voltear sí
  valen (no cambian el conteo).

**En el cuaderno:** sección 7. La función está en `datos.py`
(`capa_aumento()`).

## Paso 5. Transfer learning (no entrenar desde cero)

**Qué se hace:** se parte de una red ya entrenada con millones de fotos
(EfficientNetB0 con ImageNet), se le congela la base y solo se entrena
una cabeza clasificadora nueva para nuestras clases.

**Por qué:** entrenar una red desde cero necesita millones de fotos y
días de GPU. La base ya aprendió a ver bordes, texturas y formas; eso
sirve para cualquier foto, incluidas las de microscopio. Nosotros solo
le enseñamos la última parte: "esta forma es *Chlorella*, esta es
*Scenedesmus*".

**Si después quieres más precisión:** descongelar las últimas capas de
la base y reentrenar con una tasa de aprendizaje baja (fine-tuning).
Primero la cabeza, después el ajuste fino: en ese orden.

**En el cuaderno:** sección 8.

## Paso 6. Entrenar (y mirar las curvas)

**Qué se hace:** el modelo pasa varias veces (épocas) por el conjunto
de entrenamiento, ajustando sus pesos para acertar más. La parada
temprana detiene el entrenamiento cuando la validación deja de mejorar.

**Qué mirar:** las curvas de exactitud y pérdida de train y val.

- Si train mejora y val empeora: sobreajuste (se memoriza las fotos).
  Solución: más aumento, más fotos, dropout, menos épocas.
- Si las dos se estancan abajo: el modelo no aprende. Solución:
  revisar las etiquetas, más fotos por clase, descongelar capas.
- Si las dos mejoran juntas: va bien.

**En el cuaderno:** sección 8b (10 épocas, parada temprana con
paciencia de 3).

## Paso 7. Evaluar (la medida que importa es la de test)

**Qué se hace:** se mide el modelo con el conjunto test, que nunca vio.

**Qué mirar, además de la exactitud global:**

- **Reporte por clase:** qué clases acierta y cuáles confunde. Una
  exactitud global de 90% puede esconder una clase con 40%.
- **Matriz de confusión:** qué clases se confunden entre sí. Si dos
  especies se confunden siempre, quizá se ven igual con ese aumento,
  o las etiquetas de una de ellas están mal, o faltan fotos de esa
  clase.

**Qué hacer si una clase falla:** más fotos de esa clase, revisar sus
etiquetas, o juntar las dos clases que se confunden en una sola si en
la práctica no se distinguen.

**En el cuaderno:** sección 9.

## Paso 8. Exportar y usar el modelo

**Qué se hace:** se guardan dos archivos en
`Drive/microalgas-ia/modelos/`:

- **`modelo_microalgas.h5`:** el modelo completo de Keras, para seguir
  entrenándolo o usarlo en Python.
- **`modelo_microalgas.tflite`:** versión ligera (TensorFlow Lite) para
  correrlo en el teléfono en el campo, sin internet.
- **`clases.txt`:** la lista de clases en orden. El orden importa: la
  salida número 3 del modelo es la tercera clase de esta lista.

**En el cuaderno:** sección 10.

## Paso 9. Contar células y sacar la densidad

**Qué se hace:** para contar no hace falta red neuronal al principio.
Una línea base con OpenCV basta en fotos limpias de cultivos: pasar a
gris, suavizar, umbral adaptativo, buscar contornos y filtrar por área.

**La calibración es lo que convierte el conteo en densidad:**

```
densidad (células/mL) = células contadas / volumen fotografiado (mL)
```

Ejemplo: 120 células en un cuadro grande de la cámara de Neubauer
(0.0001 mL) = 1,200,000 células/mL. Sin el volumen, el número no
significa nada.

**Cómo calibrar en la práctica:** cuenta a mano (Neubauer) las células
de una foto, corre la función sobre esa misma foto y ajusta `area_min`,
`area_max` y el umbral hasta que el conteo automático coincida con el
tuyo. Anota esos valores: son parte del método.

**En el cuaderno:** sección 11. Las funciones están en `datos.py`
(`contar_celulas_opencv()`, `densidad_celular()`).

---

## Cómo adaptar la receta a otros modelos

La receta es la misma; lo que cambia son los datos y la última capa.

| Quiero... | Qué cambia |
|---|---|
| Otros organismos (diatomeas, cianobacterias, zooplancton) | Las fotos y las carpetas por clase. El resto igual. |
| Clasificar el **estado** del cultivo (exponencial, estacionario, estresado) en vez de la especie | Las clases son estados, no especies. Las fotos se etiquetan por estado. |
| Contar por clase en una foto con varias especies | Ya no basta clasificar la foto entera: hace falta **detectar** (YOLO) con las cajas de EMDS-7. Es el paso siguiente del proyecto. |
| Sondear un cuerpo de agua (río, laguna) | La foto del agua no muestra células: ahí el modelo solo detecta floración o grupo dominante por el color. Para especies hace falta muestra al microscopio. |
| Más clases de plancton marino | Sumar WHOI-Plankton (descarga manual) como más carpetas por clase. |

**La regla que no cambia:** el modelo solo reconoce lo que ha visto
entrenando. Cada clase nueva necesita sus fotos, sus etiquetas y su
medida en test.

## Errores comunes

1. **Pocas fotos por clase.** Con 10 fotos no se aprende una clase.
   Mínimo práctico: decenas; mejor, cientos.
2. **Fotos sin etiquetar.** Sin etiqueta, la foto no entra al
   entrenamiento.
3. **Mezclar fotos del mismo cultivo en train y test.** La medida sale
   inflada: el modelo ya vio ese cultivo.
4. **No anotar el aumento ni el volumen.** Sin esos datos, el conteo no
   se convierte en densidad.
5. **Evaluar con train.** La exactitud de entrenamiento no dice si el
   modelo sirve: la que importa es la de test.
6. **Recortar al aumentar para conteo.** Cambia el número de células
   de la foto.

## Mapa: cada paso en el cuaderno

| Paso | Sección del cuaderno |
|---|---|
| 1. Fotos | 4. Tus fotos |
| 2. Carpetas por clase | 3. Recortes, 4. Tus fotos |
| 3. train/val/test | 6. Particiones |
| 4. Aumento | 7. Aumento de datos |
| 5. Transfer learning | 8. Modelo |
| 6. Entrenar | 8b. Entrenar |
| 7. Evaluar | 9. Evaluación |
| 8. Exportar | 10. Exportar el modelo |
| 9. Contar y densidad | 11. Densidad celular |
