import argparse
import torch
from entrenamiento import train
from torch.utils.data import TensorDataset,DataLoader
from models import get_model
from datasets import coreset
import warnings
from datasets import datosYred
import os
parser=argparse.ArgumentParser(description="realizar entrenamientos para comprobar el funcionamiento del destilado.")
parser.add_argument(
    "--carpetaAnterior",
    type=str,
    help="carpeta del archivo Mejor.pt y/o hiperDest.pt a utilizar, requerida si se va entrenar con datos destilados."
)
parser.add_argument(
    "--dispositivo",
    type=int,
    default=-1,
    help="Indice del dispositivo en el que se realizará el destilado, si es negativo se usará la CPU."
)
parser.add_argument(
    "--semilla",
    type=int,
    default=18,
    help="semilla pseudoaleatoria para inicializar los pesos."
)
parser.add_argument(
    "--tipoDatos",
    type=str,
    choices=["reales","destilados","coreset"],
    default="destilados",
    help="Especificar si se realizará el entrenamiento con los datos reales (archivo images_all.pt y labels_all.pt) o destilados (imgs.pt y label_syn.pt)"
)
parser.add_argument(
    "--destino",
    type=str,
    help="Carpeta en la que se guardarán los registros de este entrenamiento, por defecto se escogerá la carpeta anterior"
)
parser.add_argument(
    "--modelo",
    type=str,
    choices=["nn","hamida","chen","li","hu"],
    help="Nombre del modelo a usar (para tipoDatos coreset y reales), también se puede especificar por medio de un archivo hiperDest.pt pero si se especifica aquí tendrá prioridad."
)
parser.add_argument(
    "--ipc",
    type=int,
    help="indices por clase (para tipoDatos coreset solamente), también se puede especificar por medio de un archivo hiperDest.pt; pero si se especifica aquí tendrá prioridad."
)
parser.add_argument("--epocas",type=int,default=100,help="Cantidad de epocas, también se puede especificar por medio de un archivo hiperDest.pt; pero si se especifica aquí tendrá prioridad.")
parser.add_argument(
    "--guardar",
    type=bool,
    help="decide si guardar o no registros en disco de este entrenamiento"
)
#definir en que carpeta se almacenarán los registros de este entrenamiento
carpeta="resultados/"
if parser.parse_args().guardar:
    destino=parser.parse_args().carpetaAnterior if parser.parse_args().destino==None else parser.parse_args().destino
    if destino==None:
        exit("Se especificó que se guardaran registros pero no se especificó un nombre para la carpeta en la que se deben guardar")
    elif not os.path.isdir(carpeta+destino):
        os.mkdir(carpeta+destino)
        print("Se creó la carpeta",carpeta+destino)
    print("Los registros se guardarán en",carpeta+destino)
    destino=carpeta+destino+'/'
else:
    if parser.parse_args().destino==None:
        print("No se guardarán registros de este entrenamiento.")
    else:
        warnings.warn("Se especificó una carpeta de destino pero la opción guardar se especificó cómo falsa; por lo tanto no se guardará ningún registro.")
    destino=None
if parser.parse_args().carpetaAnterior==None:
    exit("se necesita especificar una carpeta para realizar este entrenamiento.")
else:
    ruta_anterior=carpeta+parser.parse_args().carpetaAnterior+'/'
hiperDest=torch.load(ruta_anterior+"hiperDest.pt")if os.path.isfile(ruta_anterior+"hiperDest.pt")else None
dispositivo="cpu"if parser.parse_args().dispositivo<0 else f"cuda{parser.parse_args().dispositivo}"
#definir nombre del modelo a entrenar
if parser.parse_args().modelo==None:
    if hiperDest==None:
        exit("No se conoce el nombre del modelo a entrenar")
    modelo=hiperDest["modelo"]
else:
    modelo=parser.parse_args().modelo
#definir nombre del conjunto
if parser.parse_args().conjunto==None:
    if hiperDest==None:
        exit("No se conoce el nombre del conjunto.")
    conjunto=hiperDest["conjunto"]
else:
    conjunto=parser.parse_args().conjunto
#carguar datos de entrenamiento, validación y prueba
if parser.parse_args().tipoDatos=="destilados":
    _,test_loader,val_loader,red,optimizador_red,criterion,hiperparametros=
    img=torch.load(
        ruta_anterior+"Mejor.pt",
        map_location=dispositivo
    )
    etq=torch.repeat_interleave(torch.arange(hiperparametros["n_classes"]),ipc)
    

#cargar y actualizar hiperparametros (los cambios no se sobreescribiran en hiperparametros.pt)
hiperDest
print("Algoritmo ejecutándose en",hiperparametros["device"],"\n\n")
#cargua de los datos de entrenamiento
if parser.parse_args().tipoDatos=="reales":
    carguador=torch.load(ruta_anterior+"train_loader.pt",map_location=hiperparametros["device"])
else:
    if parser.parse_args().tipoDatos=="destilados":
        img=torch.load(
            ruta_anterior+"Mejor.pt",map_location=hiperparametros["device"]
        )
        etq=torch.load(
            ruta_anterior+"label_syn.pt",map_location=hiperparametros["device"]
        )
    else:#coreset
        img,etq=coreset(
            torch.load(ruta_anterior+"images_all.pt",map_location=hiperparametros["device"]),
            torch.load(ruta_anterior+"indices_class.pt",map_location=hiperparametros["device"]),
            torch.load(ruta_anterior+"labels_all.pt",map_location=hiperparametros["device"]),
            etq_cor=torch.load(ruta_anterior+"label_syn.pt",map_location=hiperparametros["device"])
        )
    carguador=DataLoader(
        TensorDataset(img,etq),
        batch_size=hiperparametros["batch_size"],
        shuffle=True
    )
    del img,etq
#carga del modelo y optimizadores
#hiperparametros["n_classes"]=hiperparametros["n_classes"]+1
red,optimizador,criteiron,_=get_model(hiperparametros["model"],hiperparametros["device"],**hiperparametros)
print("Iniciando entrenamiento con datos",parser.parse_args().tipoDatos)
red,perdida,accEnt,accVal,accTest=train(
    red,
    optimizador,
    criteiron,
    carguador,
    parser.parse_args().epocas,
    torch.load(ruta_anterior+"test_loader.pt",map_location=hiperparametros["device"]),
    torch.load(ruta_anterior+"val_loader.pt",map_location=hiperparametros["device"]),
    hiperparametros["device"])
for variable,archivo in zip([red,perdida,accEnt,accVal],
                            ["redEntrenada","perdida","accTrain","accVal"]):
    torch.save(variable,carpeta+carpetaDestino+f"/{archivo}Datos{parser.parse_args().tipoDatos}.pt")
#guardar accuracy de testeo
with open(carpeta+carpetaDestino+"/accTestDatosdestilados.txt", "w") as txtfile:
    print("Accuracy de testeo para datos destilados: {}".format(accTest), file=txtfile)