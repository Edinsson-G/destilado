for modelo in {hamida,nn,chen}
do
	for conjunto in {IndianPines,PaviaC,PaviaU,Botswana}
	do
		for ipc in {10,20,30,40,50}
		do
			
			anterior="Modelo ${modelo} conjunto ${conjunto} ipc ${ipc}"
			if [ $modelo == hamida ]
			then
				anterior="${anterior} lr 0.001"
			fi
			if [ -d "resultados/$anterior" ]
			then
				destino="prueba ${anterior} 1000 epocas"
				#if [ ! -d "resultados/${destino}" ] || [ ! -e "resultados/${destino}/acc_test_primer" ]
				#then
				echo $anterior
				python3 comprobarDestilado.py --modelo ${modelo} --carpetaAnterior "$anterior" --carpetaDestino "${destino}" --dispositivo $1 --conjunto ${conjunto}
				#fi
			fi
		done
	done
done