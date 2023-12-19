import argparse
import torch
from entrenamiento import train
from torch.utils.data import TensorDataset,DataLoader
from models import get_model
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
    choices=["reales","destilados"],
    default="destilados",
    help="Especificar si se realizará el entrenamiento con los datos reales (archivo images_all.pt y labels_all.pt) o destilados (imgs.pt y label_syn.pt)"
)
parser.add_argument(
    "--indice",
    type=int,
    help="En caso que se quiera hacer el entrenamiento con imágenes sintéticas y el archivo imgs.pt contenga una lista de ellas (situación que se da en el caso que se haya especificado el atributo historial como True al momento del destilado), se sellecionarán las imágenes correspondientes a este índice"
)
parser.add_argument(
    "--carpetaDestino",
    type=str,
    help="Carpeta en la que se guardarán los registros de este entrenamiento"
)
parser.add_argument("--epocas",type=int,default=100,help="Cantidad de epocas.")
parser.add_argument(
    "--tecAumento",
    type=str,
    choices=["ruido","escalamiento","ninguno"],
    default="ruido",
    help="Técnica que se utilizará para hacer aumento de datos. Ruido consiste en sumar ruido pseudoaleatorio uniformemente distribuido en el intervalo [-0.05,0.05]"
)
parser.add_argument(
    "--factAumento",
    type=int,
    default=2,
    help="Cantidad de datos nuevos a generar a partir de cada dato de entrenamiento"
)
carpeta="resultados/"
ruta_anterior=carpeta+parser.parse_args().carpetaAnterior+'/'
if parser.parse_args().carpetaDestino==None:
    carpetaDestino=parser.parse_args().carpetaAnterior
else:
    carpetaDestino=parser.parse_args().carpetaDestino
#cargar y actualizar hiperparametros (los cambios no se sobreescribiran en hiperparametros.pt)
hiperparametros=torch.load(ruta_anterior+"hiperparametros.pt")
hiperparametros["cuda"]=parser.parse_args().dispositivo
hiperparametros["device"]=torch.device("cpu" if parser.parse_args().dispositivo<0 else "cuda:"+str(parser.parse_args().dispositivo))
torch.set_default_device(hiperparametros["device"])
torch.manual_seed(parser.parse_args().semilla)
print("Algoritmo ejecutándose en",hiperparametros["device"],"\n\n")
#cargua de los datos de entrenamiento
if parser.parse_args().tipoDatos=="destilados":
    img=torch.load(ruta_anterior+"imgs.pt",map_location=hiperparametros["device"])
    if type(img)==list:
        #en el destilado el parámetro historial era True
        if parser.parse_args().indice==None:
            #se escoge el indice de la iteración con menor pérdida
            perdida=torch.load(ruta_anterior+"histPerdida.pt")
            img=img[perdida.index(min(perdida))]
            del perdida
        else:
            img=img[parser.parse_args().indice]
    etiquetas=torch.load(ruta_anterior+"label_syn.pt",map_location=hiperparametros["device"])
else:
    img=torch.load(ruta_anterior+"images_all.pt",map_location=hiperparametros["device"])
    etiquetas=torch.load(ruta_anterior+"labels_all.pt",map_location=hiperparametros["device"])
img.requires_grad_(False)
#ejecutar aumento si aplica
if parser.parse_args().tecAumento=="ruido":
    from datos import adicion
    img,etiquetas=adicion(img,parser.parse_args().factAumento,etiquetas)
elif parser.parse_args().tecAumento=="escalamiento":
    from datos import noAdicion
    img,etiquetas=noAdicion(img,torch.rand(parser.parse_args().factAumento),"escalamiento",etiquetas)
maxi=torch.max(img)
if maxi>1:
    img=img/maxi
#mostrar y guaradar estadisticos
with open(f"EstadisticosDatos{parser.parse_args().tipoDatos}.txt", "w") as archivotxt:
    print(f"media: {torch.mean(img)}\ndesviación:{torch.std(img)}",file=archivotxt)
#carga del modelo y optimizadores
hiperparametros["n_classes"]=hiperparametros["n_classes"]+1
red,optimizador,criteiron,_=get_model(hiperparametros["model"],hiperparametros["device"],**hiperparametros)
print("Iniciando entrenamiento con datos",parser.parse_args().tipoDatos)
for variable,archivo in zip(
    train(red,
          optimizador,
          criteiron,
          DataLoader(TensorDataset(img,etiquetas),
                    batch_size=hiperparametros["batch_size"],
                    shuffle=True,
                    num_workers=0,
                    generator=torch.Generator(device=hiperparametros["device"])),
          parser.parse_args().epocas,
          torch.load(ruta_anterior+"test_loader.pt",map_location=hiperparametros["device"]),
          torch.load(ruta_anterior+"val_loader.pt",map_location=hiperparametros["device"]),
          hiperparametros["device"]),
    ["redEntrenada","perdida","accTrain","accVal","accTest"]
):
    torch.save(variable,carpeta+carpetaDestino+f"/{archivo}Datos{parser.parse_args().tipoDatos}.pt")