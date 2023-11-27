for ipc in {10,20,30,40,50}
do
    #comprobar si hay que continuar el destilado o iniciarlo
    destino="Modelo ${1} conjunto ${2} ipc $ipc lr $4"
    if [ -d "resultados/$destino" ]
    then
        echo "Reanudando entrenamiento..."
        python3 ejecutarDestilado.py --dispositivo $3 --carpetaAnterior "$destino" --carpetaDestino "$destino" --iteraciones 500
        echo "Entrenamiento en $destino ya terminado"
    else
        echo "iniciando nuevo entrenamiento..."
        python3 ejecutarDestilado.py --modelo $1 --dataset $2 --dispositivo $3 --semilla 0 --carpetaDestino "$destino" --ipc $ipc --lrImg $4 --iteraciones 500
        echo "terminado entrenamiento en $destino "
    fi
done