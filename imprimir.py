import torch
import os
import pandas as pd
def impresion_segura(ruta,archivo,indice=None):
    try:
        valor=torch.load(ruta+'/'+archivo)if archivo in os.listdir(ruta) else f"{ruta}/{archivo} no existe"
    except EOFError:
        valor="Archivo vacío"
    finally:
        if indice=="min":
            valor=valor[valor.index(min(valor))]
        elif indice!=None and (type(valor)==dict or type(valor)==list):
            valor=valor[indice]
        if torch.is_tensor(valor)and valor.numel()==1:
            valor=valor.item()
        return valor
resultados="resultados"
claves=["carpeta","numClases","accEntr","accTest","accVal","PerdidaMin"]
tabla={clave:[] for clave in claves}
for carpeta in os.listdir(resultados):
    ruta=resultados+'/'+carpeta
    tabla["carpeta"].append(carpeta)
    for clave,valor in zip(
        claves[1:],#el for superior ya iterará las carpetas, no hay que hacerlo aquí
        [
            impresion_segura(ruta,"hiperparametros.pt","n_classes"),
            impresion_segura(ruta,"accTrainDatosdestilados.pt",-1),
            impresion_segura(ruta,"accTestDatosdestilados.pt"),
            impresion_segura(ruta,"accValDatosdestilados.pt",-1),
            impresion_segura(ruta,"histPerdida.pt","min")
        ]
    ):
        tabla[clave].append(valor)
pd.DataFrame(tabla).to_csv(r"preliminares.csv",index=False)