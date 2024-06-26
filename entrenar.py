import argparse
import torch
from entrenamiento import train
from torch.utils.data import TensorDataset,DataLoader
from models import get_model
from datasets import coreset,datosYred,vars_all
import warnings
import os
import numpy as np
from tqdm import tqdm
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
    choices=["reales","destilados","aleatorio","herding"],
    default="destilados",
    help="Especificar si se realizará el entrenamiento con los datos reales (archivo images_all.pt y labels_all.pt), destilados (imgs.pt y label_syn.pt), coreset de muestreo aleatorio o coreset herding"
)
parser.add_argument(
    "--destino",
    type=str,
    help="Carpeta en la que se guardarán los registros de este entrenamiento, por defecto se escogerá la carpeta anterior"
)
parser.add_argument(
    "--modelo",
    type=str,
    choices=["nn","hamida","li"],
    help="Nombre del modelo a usar (para tipoDatos coreset y reales), también se puede especificar por medio de un archivo hiperDest.pt pero si se especifica aquí tendrá prioridad."
)
parser.add_argument(
    "--ipc",
    type=int,
    help="indices por clase (para tipoDatos coreset solamente), también se puede especificar por medio de un archivo hiperDest.pt; pero si se especifica aquí tendrá prioridad."
)
parser.add_argument("--epocas",type=int,default=1000,help="Cantidad de epocas, también se puede especificar por medio de un archivo hiperDest.pt, el cual tendrá prioridad")
parser.add_argument(
    "--guardar",
    type=bool,
    default=True,
    help="decide si guardar o no registros en disco de este entrenamiento"
)
parser.add_argument(
    "--conjunto",
    type=str,
    choices=["IndianPines","PaviaC","PaviaU","Botswana"]
)
parser.add_argument(
    "--coreset_escalable",
    type=bool,
    help="Si es verdadero y el tipo de datos es una técnica de coreset entonces se ejecutará de manera escalable para evitar problemas de memoria.",
    default=False
)
parser.add_argument(
    "--repeticiones",
    type=int,
    default=20,
    help="cantidad de veces que se repetirá el entrenamiento"
)
parser.add_argument(
    "--archivo",
    type=str,
    help="archivo donde estan los datos a entrenar",
    default="Mejor_perdida.pt"
)
#definir en que carpeta se almacenarán los registros de este entrenamiento
carpeta="resultados/"
if parser.parse_args().guardar:
    destino=parser.parse_args().carpetaAnterior if parser.parse_args().destino==None else parser.parse_args().destino
    if destino==None:
        destino=f"{parser.parse_args().modelo}_{parser.parse_args().conjunto}_ipc{parser.parse_args().ipc}"
    if not os.path.isdir(carpeta+destino):
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
ruta_anterior=None if parser.parse_args().carpetaAnterior==None else carpeta+parser.parse_args().carpetaAnterior+'/'
hiperDest=torch.load(ruta_anterior+"hiperDest.pt")if ruta_anterior!=None and os.path.isfile(ruta_anterior+"hiperDest.pt")else None
dispositivo="cpu"if parser.parse_args().dispositivo<0 else f"cuda:{parser.parse_args().dispositivo}"
#carguar datos de entrenamiento, validación y prueba
modelo=parser.parse_args().modelo if parser.parse_args().modelo!=None else hiperDest["modelo"]if hiperDest!=None else exit("No se conoce el nombre del modelo a entrenar")
conjunto=parser.parse_args().conjunto if parser.parse_args().conjunto!=None else hiperDest["conjunto"] if hiperDest!=None else exit("No se conoce el nombre del conjunto.")
semilla=hiperDest["semilla"]if hiperDest!=None else 0
torch.manual_seed(0)
np.random.seed(0)
dst_train,test_loader,val_loader,red,_,_,hiperparametros=datosYred(
modelo,conjunto,parser.parse_args().dispositivo)
torch.manual_seed(semilla)
np.random.seed(semilla)
del semilla
if parser.parse_args().tipoDatos=="reales":
    carguador=DataLoader(
        dst_train,
        batch_size=hiperparametros["batch_size"],
        shuffle=True
    )
else:
    etq=torch.repeat_interleave(
        torch.arange(hiperparametros["n_classes"]),
        parser.parse_args().ipc if parser.parse_args().ipc!=None else hiperDest["ipc"]if hiperDest!=None else exit("No se conoce el ipc.")
    )
    if parser.parse_args().tipoDatos=="destilados":
        #_,test_loader,val_loader,red,optimizador_red,criterion,hiperparametros=
        if ruta_anterior==None:
            exit("Para entrenar datos destilados debe especificarla carpeta en la que se encuentran alojados (archivo Mejor.pt).")
        if not os.path.isfile(ruta_anterior+parser.parse_args().archivo):
            exit("no se encuentran los datos destilados (Mejor.pt) en "+ruta_anterior)
        del dst_train
        img=torch.load(
            ruta_anterior+parser.parse_args().archivo,
            map_location=dispositivo
        ).detach()
    else:#coreset
        images_all,_,indices_class=vars_all(dst_train,hiperparametros["n_classes"])
        img=coreset(images_all,indices_class,hiperDest["ipc"]if parser.parse_args().ipc==None else parser.parse_args().ipc,parser.parse_args().tipoDatos,parser.parse_args().coreset_escalable,0)
    carguador=DataLoader(
            TensorDataset(img,etq),
            batch_size=hiperparametros["batch_size"],
            shuffle=True
        )
    #graficar(img,etq)
print("Algoritmo ejecutándose en",dispositivo,"\n\n")
print("Iniciando entrenamiento con datos",parser.parse_args().tipoDatos)
#obtener red con los pesos definidos por la semilla especificada
torch.manual_seed(parser.parse_args().semilla)
#accuracies de testeo
accs_test=torch.empty(parser.parse_args().repeticiones)
for i in tqdm(range(parser.parse_args().repeticiones)):
    red,optimizador,criterion,_=get_model(modelo,hiperparametros["cuda"],**hiperparametros)
    red,perdida,accEnt,accVal,accTest=train(
        red,
        optimizador,
        criterion,
        carguador,
        parser.parse_args().epocas,
        test_loader,
        val_loader,
        hiperparametros["cuda"]
    )
    accs_test[i]=accTest
    tqdm.write(f"accuracy test experimento{i}: {accTest}")
if destino!=None:
    for variable,archivo in zip([red.state_dict(),perdida,accEnt,accVal,accs_test],
                                ["pesos","perdida","accTrain","accVal","accs_test"]):
        torch.save(variable,destino+f"/{archivo}Datos{parser.parse_args().tipoDatos}.pt")
    #guardar accuracy de testeo
    with open(destino+"accTestDatos"+parser.parse_args().tipoDatos+".txt", "w") as txtfile:
        print(f"Accuracy de testeo: {torch.mean(accs_test)}+-{torch.std(accs_test)}", file=txtfile)
print("acuracy promedio:",torch.mean(accs_test).item(),"+-",torch.std(accs_test).item())