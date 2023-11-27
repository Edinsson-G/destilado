import torch
import os
for modelo in ["nn","hamida"]:
    for ipc in range(10,60,10):
        for conjunto in ["IndianPines","PaviaC","PaviaU","Botswana"]:
            carpeta="resultados"
            subcarpeta="Modelo "+modelo+" conjunto "+conjunto+" ipc "+str(ipc)
            if modelo=="hamida":
                subcarpeta=subcarpeta+" lr 0.001"
            if "prueba "+subcarpeta+" 1000 epocas" in os.listdir(carpeta):
            #if "prueba "+subcarpeta+" 1000 epocas" in os.listdir(carpeta)and len(torch.load(carpeta+'/'+subcarpeta+"/perdida"))>=500:
                print("\n\n cantidad de iteraciones:",len(torch.load(carpeta+'/'+subcarpeta+"/perdida")))
                subcarpeta=subcarpeta+" 1000 epocas"
                print(subcarpeta)
                print("datos destilados:")
                print("accuracy de entrenamiento",torch.load(carpeta+"/prueba "+subcarpeta+"/acc_train"))
                print("accuracy de testeo",torch.load(carpeta+"/prueba "+subcarpeta+"/acc_test"))
                print("firmas reales muestreadas")
                print("accuracy de entrenamiento",torch.load(carpeta+"/prueba "+subcarpeta+"/acc_ent_primer"))
                print("accuracy de testeo",torch.load(carpeta+"/prueba "+subcarpeta+"/acc_test_primer"))