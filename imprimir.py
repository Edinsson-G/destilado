import torch
import os
import pandas as pd
def impresion_segura(ruta,archivo,indice=None):
    try:
        valor=torch.load(ruta+'/'+archivo)if archivo in os.listdir(ruta) else "error"
    except EOFError:
        valor="error"
    finally:
        if indice=="min":
            valor=valor[valor.index(min(valor))]
        elif indice!=None and (type(valor)==dict or type(valor)==list):
            valor=valor[indice]
        if torch.is_tensor(valor)and valor.numel()==1:
            valor=valor.item()
        return valor
resultados="resultados"
claves=["modelo","conjunto","numClases","ipc","lr","aumento","accEntr","accTest","PerdidaMin"]
tabla={clave:[] for clave in claves}
for carpeta in os.listdir(resultados):
    ruta=resultados+'/'+carpeta
    #dividir el nombre de la carpeta en palabras para adquirir nombre del modelo, lr, ipc, etc
    palabras=carpeta.split(' ')
    valores=[
        palabras[1],#modelo
        palabras[3],#conjunto de datos
        impresion_segura(ruta,"hiperparametros.pt","n_classes"),#número de clases
        int(palabras[5]),#ipc
        float(palabras[9]),#ritmo de aprendizaje
        palabras[11],#tipo de aumento
        impresion_segura(ruta,"accTrainDatosdestilados.pt",-1),#accuracy de entrenamiento
        impresion_segura(ruta,"accTestDatosdestilados.pt"),#accuracy de testeo
        impresion_segura(ruta,"histPerdida.pt","min")#minimo valor de perdida
    ]
    if "error"not in valores:
        for clave,valor in zip(claves,valores):
            tabla[clave].append(valor)
pd.DataFrame(tabla).sort_values(["modelo","conjunto","ipc","lr","aumento"]).to_csv(r"preliminares.csv",index=False)