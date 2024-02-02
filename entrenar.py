import argparse
import torch
from entrenamiento import train
from torch.utils.data import TensorDataset,DataLoader
from models import get_model
from datasets import coreset
from utiles import aumento
parser=argparse.ArgumentParser(description="realizar entrenamientos para comprobar el funcionamiento del destilado.")
parser.add_argument(
    "--carpetaAnterior",
    type=str,
    required=True,
    help="Nombre de la carpeta en la que se almacenan los datos que se utilizarán para el entrenamiento, debe estar dentro de la carpeta resultados."
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
    "--carpetaDestino",
    type=str,
    help="Carpeta en la que se guardarán los registros de este entrenamiento"
)
parser.add_argument("--epocas",type=int,default=100,help="Cantidad de epocas.")
carpeta="resultados/"
ruta_anterior=carpeta+parser.parse_args().carpetaAnterior+'/'
carpetaDestino=parser.parse_args().carpetaAnterior if parser.parse_args().carpetaDestino==None else parser.parse_args().carpetaDestino
#cargar y actualizar hiperparametros (los cambios no se sobreescribiran en hiperparametros.pt)
hiperparametros=torch.load(ruta_anterior+"hiperparametros.pt")
hiperparametros["cuda"]=parser.parse_args().dispositivo
hiperparametros["device"]=torch.device("cpu" if parser.parse_args().dispositivo<0 else "cuda:"+str(parser.parse_args().dispositivo))
torch.manual_seed(parser.parse_args().semilla)
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