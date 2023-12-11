#dispositivo $1
for carpeta in resultados/*
do
    python3 comprobarDestilado.py --carpetaAnterior "$(basename "$carpeta")" --dispositivo $1
done