#$1: red
#$2: conjunto de datos
#$3: dispositivo
source entorno/bin/activate
for ipc in {10,20,30,40,50}
do
    for lr in {1,0.1,0.01,0.001,0.0001}
    do
        for tecnica in {escalamiento,ruido,None}
        do
            for factorAumento in {1,2,3,4}
            do
                python3 ejecutarDestilado.py --modelo $1 --conjunto $2 --dispositivo $3 --ipc $ipc --lrImg $lr --tecAumento $tecnica --factAumento $factorAumento --reanudar True
            done
        done
    done
done