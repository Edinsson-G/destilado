for lr in {0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001,0.000000001,0.0000000001}
do
    python3 ejecutarDestilado.py --modelo nn --conjunto IndianPines --dispositivo $1 --ipc 10 --lrImg $lr
done
echo Evaluando resultados
for carpetaAnterior in resultados/*
do
    echo evaluando $(basename "$carpetaAnterior" )
    python3 comprobarDestilado.py --carpetaAnterior "$(basename "$carpetaAnterior" )" --dispositivo $1
done