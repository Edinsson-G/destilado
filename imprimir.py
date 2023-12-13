import torch
import os
import pandas as pd
def impresion_segura(ruta,archivo,indice=None):
    valor=torch.load(ruta+'/'+archivo)if archivo in os.listdir(ruta) else f"{ruta}/{archivo} no existe"
    if indice!=None and (type(valor)==dict or type(valor)==list):
        valor=valor[indice]
    return valor
resultados="resultados"
for carpeta in os.listdir(resultados):
    ruta=resultados+'/'+carpeta
    print("carpeta:",carpeta)
    print("número de clases:",impresion_segura(ruta,"hiperparametros.pt"),"n_classes")
    print("accuracy de entrenamiento:",impresion_segura(ruta,"accTrainDatosdestilados.pt"),-1)
    print("accuracy de validación:",impresion_segura(ruta,"accValDatosdestilados.pt"),-1)
    print("accuracy de testeo:",impresion_segura(ruta,"accTestDatosdestilados.pt"))