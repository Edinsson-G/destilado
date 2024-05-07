# Destilado de bases de datos de imágenes hiperespectrales basado en aprendizaje profundo para la solución de problemas de clasificación
La presente es la implementación de un algoritmo para destilar bases de datos de clasificación de imágenes hiperespectrales en base a la coincidencia de distribución.

## Configuración de entorno
Todas las librerías de terceros necesarias para utilizar este repositorio están en el archivo `librerias.txt`, puede instalarlas desde una terminal Linux con el siguiente comando:
```
pip install -r librerias.txt
```

## Destilar datos
Para iniciar el destilado de un determinado conjunto de ejemplo debe ejecutar el archivo `destilar.py` desde una terminal. Las opciones mas importantes que se le pueden pasar a este archivo son las siguientes:

- `--modelo`: Es el nombre del modelo que se quiere utilizar para el destilado. Las opciones disponibles son "nn", "hamida" y "li"[^1].
- `--conjunto`: Nombre de la base de datos de clasificación a destilar. Las opciones disponibles por defecto son "IndianPines", "Botswana" y "PaviaU"[^1].
- `--lrImg`: Valor de la tasa de aprendizaje para la optimización de los datos sintéticos.
- `--ipc`: indices por clase o muestras por clase. Determina el tamaño del conjunto de datos sintético en términos de la cantidad de muestras que este tendría por cada clase.
- `--inicialización`: Indica cómo se inicilializarán los datos sintéticos. Las opciones disponibles por defecto son "aleatorio", para inicializar el conjunto con muestras aleatorias del conjunto de entrenamiento original; "herding", para inicializarlo con kernel herding y "ruido", para inicializar los datos con ruido.
- `--carpetaDestino`: nombre de la carpeta en la que se guardarán los registros de este destilado, la cual podrá encontrarse dentro de la carpeta `resultados`, algunos de ellos son:
  - `acc_list.pt`: es el historial de accuracies de validación obtenidos a lo largo del destilado.
  - `hist_perdida.pt`: historial del valor de pérdida de los datos sintéticos a lo largo de las iteraciones.
  - `Mejor_perdida.pt`: datos sintéticos en la iteración de menor valor de pérdida.
- `--carpeta_anterior`: en caso de querer reanudar un destilado previo indicar el nombre de la carpeta en la que se alojaron sus registros, naturalmente estos archivos deben estar dentro de la carpeta resultados.
- `--iteraciones`: cantidad de iteraciones a realizar.
- `--semilla`: semilla pseudoaleatoria a utilizar, por defecto es 0.
- `--dispositivo` indice del dispositivo cuda en el que se realizará el destilado, si es un entero negativo este se realizará en la CPU. Por defecto es -1.

## Entrenar modelos
Con el fin de validar los datos sintetizados puede utilizar el archivo `entrenar.py` para entrenar los diferentes modelos de red neuronal. Para ello se le pueden pasar los diferentes argumentos:
- `--tipoDatos`: especificar si se desea entrenar con datos destilados, kernel herding o coreset aleatorio; para lo cual se le deberá pasar como valor de este argumento "destilados", "herding" o "aleatorio" respectivamente. Por defecto su valor es "destilados".
- `--carpetaAnterior`: Obligatorio si el argumento `--tipoDatos` es "destilados" ya que es el nombre de la carpeta en la cual se alojan los registros de un destilado anterior del cual se obtendrán los datos sintéticos a utilizar en tal caso e hiperparámetros tales cómo muestras por clase, nombre del modelo y nombre del conjunto de datos los cuales también pueden especificarse manualmente con los argumentos `--ipc`, `--modelo`, `--conjunto` respectivamente.
- `--repeticiones`: cantidad de veces que se repetirá el entrenamiento (cada vez con una inicialización diferente de los pesos del modelo).
- `--destino`: nombre de la carpeta en la que se guardarán los registros de este entrenamiento la cual se alojará en la carpeta resultados, si no se especifica este argumento pero si el argumento `--carpetaDestino` entonces los registros se guardarán allí. Entre los archivos que se generan está la lista de valores de accuracy de testeo obtenido en cada repetición cuyo nombre inicia con `accs_Datos...` y un archivo de extensión `.txt` con la media y la desviación estándar de dichos accuracies.
- `--dispositivo`: exactamente la misma función que en el archivo `destilar.py`.
- `--semilla`: Función similar a la del archivo `destilar.py`. Por defecto es 18.

Para ver una ayuda y descripción mas detallada de los argumentos de cada arhcivo puede ejecutar los siguientes comandos en una terminal de Linux según corresponda al archivo del cual desea obtener información:
```
python3 destilar.py --help
python3 entrenar.py --help
```

## Referencias
[^1]: N. Audebert, B. Le Saux and S. Lefevre, "Deep Learning for Classification of Hyperspectral Data: A Comparative Review," in IEEE Geoscience and Remote Sensing Magazine, vol. 7, no. 2, pp. 159-173, June 2019.
