import os
import torch
import copy
from models import get_model
from utiles import *
import argparse
import numpy as np
from datasets import datosYred,vars_all,coreset
from torch.utils.data import DataLoader,TensorDataset
import warnings
from tqdm import tqdm
from entrenamiento import train
from torchinfo import summary
import sys
#configurar argumentos
parser=argparse.ArgumentParser(description="Destilar imagenes hiperespectrales")
parser.add_argument(
    "--modelo",
    type=str,
    choices=["nn","hamida","chen","li","hu"],
    default="nn",
    help="Nombre del modelo de red neuronal a utilizar en el destilado"
)
parser.add_argument(
    "--conjunto",
    type=str,
    choices=["PaviaC","PaviaU","IndianPines","Botswana"],
    default="IndianPines",
    help="Nombre del conjunto de datos a destilar"
)
parser.add_argument(
    "--dispositivo",
    type=int,
    default=-1,
    help="Índice del dispositivo de procesamiento en el que se ejecutará el algoritmo, si es negativo se ejecutará en la CPU."
)
parser.add_argument("--semilla",type=int,help="Semilla pseudoaleatorio a usar.",default=0)
parser.add_argument(
    "--historial",
    type=bool,
    default=False,
    help="Si es verdadero se almacenará el historial de cambios de las firmas destiladas a lo largo de todas las iteraciones en el archivo image_syn.pt."
)
parser.add_argument(
    "--carpetaAnterior",
    type=str,
    help="Para reanudar un destilado anterior, debe especificar el nombre de la carpeta que contiene los registros de dicho destilado."
)
parser.add_argument(
    "--ipc",
    type=int,
    default=10,
    help="Imagenes por clase, obligatorio para iniciar un destilado nuevo"
)
parser.add_argument(
    "--lrImg",
    type=float,
    default=0.0001,
    help="Tasa de aprendizaje del algoritmo de destilamiento, requerido solo en caso de iniciar un nuevo destilado."
)
parser.add_argument(
    "--iteraciones",
    type=int,
    help="Cantidad total de iteraciones a realizar.",
    default=500
)
parser.add_argument(
    "--factAumento",
    type=int,
    default=20,
    help="Factor de aumento, cantidad de muestras nuevas a generar por cada ejemplo en el aumento de datos."
)
parser.add_argument(
    "--tecAumento",
    type=str,
    choices=["ruido","escalamiento","potencia","None"],
    default="escalamiento",
    help="Tecnica con la que se hará el aumento de datos, obligatoria si fracAumento es positiva"
)
parser.add_argument(
    "--inicializacion",
    type=str,
    choices=["ruido","aleatorio","herding"],
    help="Inicialización de las imágenes destiladas, aleatorio significa seleccionar aleatoriamente muestras del conjunto de entrenamiento original y aleatoriedad significa inicializarlas siguiendo una distribución uniforme con números entre 0 y 1.",
    default="aleatorio"
)
parser.add_argument(
    "--carpetaDestino",
    type=str,
    help="Nombre de la subcarpeta en la que se almacenarán los resultados del algoritmo, debe estar en el directorio /resultados, si no existe entoces se creará."
)
parser.add_argument(
    "--reanudar",
    type=bool,
    default=False,
    help="En caso que no se haya especificado una carpeta anterior verificar si la carpeta destino existe para reanudar el destilado que contiene."
)
parser.add_argument(
    "--epocas",
    type=int,
    default=1000,
    help="cantidad de epocas en etapa de validación."
)
parser.add_argument(
    "--coreset_escalable",
    type=bool,
    default=False,
    help="En caso de inicializar las muestras utilizando un método de coreset y ser verdadero se utilzará un algoritmo escalable para evitar problemas de memoria."
)
#torch.set_default_device(device)
if (parser.parse_args().factAumento>0) and (parser.parse_args().tecAumento==None):
    exit("Se especificó un factor de aumento de aumento pero no un método de aumento.")
elif parser.parse_args().factAumento<0:
    exit("El factor de aumento no debe ser negativo.")
elif (parser.parse_args().tecAumento=="ruido") and (parser.parse_args().factAumento==0):
    warnings.warn("Se especificó la técnica de aumento de ruido; sin embargo el factor de aumento es 0, por lo que no se realizará ningún aumento.")
device=torch.device("cpu" if parser.parse_args().dispositivo<0 else "cuda:"+str(parser.parse_args().dispositivo))
print("Algoritmo ejecutandose en",device)
#crear carpeta en la que se guardarán los resultados
carpeta="resultados"
if carpeta not in os.listdir('.'):
    os.mkdir(carpeta)
#definir carpetas anteriores y destino
destino='_'.join([parser.parse_args().modelo,parser.parse_args().conjunto,f"ipc{parser.parse_args().ipc}"]) if parser.parse_args().carpetaDestino==None else parser.parse_args().carpetaDestino
carpetaAnterior=parser.parse_args().carpetaAnterior
if carpetaAnterior==None and parser.parse_args().reanudar and destino in os.listdir(carpeta):
    carpetaAnterior=destino
if destino not in os.listdir(carpeta):
    os.mkdir(carpeta+'/'+destino)
ruta=carpeta+'/'+destino+'/'
if carpetaAnterior!=None:
    ruta_anterior=carpeta+'/'+carpetaAnterior+'/'
del carpeta,destino
hiperDest=vars(parser.parse_args())if carpetaAnterior==None else torch.load(ruta_anterior+"hiperDest.pt")
#obtener modelos, optimizadores y datos
torch.manual_seed(hiperDest["semilla"])
np.random.seed(hiperDest["semilla"])
#carguar datos, modelo optimizador y pérdida de entrenamiento
if parser.parse_args().epocas==0:
    #no se hará validación
    dst_train,_,_,primer_red,_,_,hiperparametros=datosYred(hiperDest["modelo"],hiperDest["conjunto"],parser.parse_args().dispositivo)
else:
    dst_train,_,val_loader,primer_red,optimizador_red,criterion,hiperparametros=datosYred(hiperDest["modelo"],hiperDest["conjunto"],parser.parse_args().dispositivo)
    #pesos originales
    orig_pesos=copy.deepcopy(primer_red.state_dict())
summary(primer_red)
if parser.parse_args().epocas==0:
    del primer_red
images_all,labels_all,indices_class=vars_all(dst_train,hiperparametros["n_classes"])
images_all=images_all.to(device)
labels_all=labels_all.to(device)
del dst_train
for c in range(hiperparametros["n_classes"]):
    print('class c = %d: %d real images'%(c, len(indices_class[c]))) # Prints how many labels are for each class
ipc=hiperDest["ipc"]
if carpetaAnterior==None:
    print("Iniciando nuevo destilado en ",ruta)
    label_syn=torch.repeat_interleave(torch.arange(hiperparametros["n_classes"]),ipc)
    if parser.parse_args().inicializacion=="ruido":
        tam=list(images_all.shape)
        tam[0]=hiperparametros["n_classes"]*ipc
        image_syn=torch.rand(tam,requires_grad=True,device=device)
        del tam
    else:
        #image_syn=coreset(images_all,indices_class,label_syn,parser.parse_args().inicializacion)
        image_syn=coreset(images_all,indices_class,ipc,parser.parse_args().inicializacion,parser.parse_args().coreset_escalable,parser.parse_args().semilla)
        image_syn.requires_grad_()
    if parser.parse_args().historial:
        historial_imagenes_sinteticas=[]
    hist_perdida=[]
    if parser.parse_args().epocas==0:#no hacer validación
        #inicializar con el valor máximo flotante soportado por esta máquina
        minPerd=sys.float_info.max
    else:
        acc_list=[]
        max_acc=-0.1
    estancamiento=0
    torch.save(hiperDest,ruta+"hiperDest.pt")
    #, momentum=0.5
    optimizer_img = torch.optim.SGD([image_syn], lr=parser.parse_args().lrImg)
    #optimizer_img=torch.optim.Adam([image_syn], lr=parser.parse_args().lrImg)
else:#se va a reanudar un entrenatiento previo
    print("Reanudando entrenamiento en",carpetaAnterior)
    torch.set_rng_state(torch.load(ruta_anterior+"tensorSemilla.pt"))
    label_syn=torch.repeat_interleave(torch.arange(hiperparametros["n_classes"]),ipc)
    if hiperDest["historial"]:
        historial_imagenes_sinteticas=torch.load(ruta_anterior+"hist_img.pt",map_location=hiperparametros["cuda"])
    image_syn=torch.load(ruta_anterior+"image_syn.pt",map_location=hiperparametros["cuda"])
    hist_perdida=torch.load(ruta_anterior+"hist_perdida.pt")
    if parser.parse_args().epocas==0 and os.path.isdir(ruta_anterior+"acc_list.pt"):
        acc_list=torch.load(ruta_anterior+"acc_list.pt")
        max_acc=max(acc_list)
    estancamiento=len(acc_list)-acc_list.index(max_acc)
    optimizer_img=torch.load(ruta_anterior+"optimizer_img.pt")
    print("Los resultados se guardarán en ",ruta)
if parser.parse_args().epocas==0:
    planificador=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_img,"min",patience=100,verbose=False)
else:
    planificador=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_img,"max",patience=100,verbose=True)
ciclo=tqdm(range(len(hist_perdida),hiperDest["iteraciones"]+1))
#definir la funcion que generará muestras nuevas para el aumento de datos
if hiperDest["tecAumento"]!="None":
    aumentador={
            "ruido":lambda tensor:tensor+torch.rand(1).item()/2.5-0.2,
            "escalamiento":lambda tensor:tensor*(torch.rand(1).item()/4+0.85),
            "potencia":lambda tensor:torch.pow(tensor,torch.ran(1).item()/4+0.85)
        }[hiperDest["tecAumento"]]
for iteracion in ciclo:
    if estancamiento>300:
        print("Finalización temprana")
        break
    ciclo.set_description_str(f"Iteración {iteracion}/{hiperDest['iteraciones']}")
    #actualizar imágenes sintéticas
    #reiniciar pesos
    net,_,_,_= get_model(hiperparametros["model"],device,**hiperparametros)
    net.train()
    perdida=torch.tensor(0.0).to(device)
    for parametros in list(net.parameters()):
        parametros.requires_grad = False
    for clase in range(hiperparametros["n_classes"]):
        #salida sin aumento
        img_real=get_images(clase,hiperparametros["batch_size"],indices_class,images_all).to(device)
        img_sin=image_syn[clase*ipc:(clase+1)*ipc]
        #aplicar aumento
        if hiperDest["tecAumento"]!="None":
            img_real=aumento(aumentador,hiperDest["factAumento"],img_real)
            img_sin=aumento(aumentador,hiperDest["factAumento"],img_sin)
        #aplicar embebido
        if hiperDest["modelo"]=="hu":
            img_real=torch.unsqueeze(img_real,1)
            img_sin=torch.unsqueeze(img_sin,1)
        salida_real=embebido(net,img_real).detach()
        output_sin=embebido(net,img_sin)
        #funcion de perdida
        perdida+=torch.sum((torch.mean(salida_real,dim=0)-torch.mean(output_sin,dim=0))**2)+torch.sum((torch.std(salida_real,dim=0)-torch.std(output_sin,dim=0))**2)
    optimizer_img.zero_grad()
    perdida.backward()
    optimizer_img.step()
    #guardar generador de números pseuadoaleatorios
    torch.save(torch.get_rng_state(),ruta+"tensorSemilla.pt")
    #guardar registros de ol
    if hiperDest["historial"]:
        historial_imagenes_sinteticas.append(copy.deepcopy(image_syn).to("cpu"))
        torch.save(historial_imagenes_sinteticas,ruta+"hist_img.pt")
    if parser.parse_args().epocas==0:
        #conservar las imágenes de la menor pérdida
        if perdida.item()<minPerd:
            torch.save(image_syn.detach(),ruta+"image_syn.pt")
            minPerd=perdida.item()
        planificador.step(perdida.item())
    else:
        #validar red
        #restablecer los pesos a los primeros generados
        primer_red.load_state_dict(orig_pesos)
        primer_red.train()
        optimizador_red.zero_grad()
        _,acc=train(
            primer_red,
            optimizador_red,
            criterion,
            DataLoader(
                TensorDataset(
                    copy.deepcopy(image_syn.detach()),
                    label_syn
                ),
                batch_size=hiperparametros["batch_size"],
                shuffle=True,num_workers=0
            ),
            test_loader=val_loader,#se usarán los datos de validación cómo si fueran los de testeo en este caso
            device=device,
            epoch=parser.parse_args().epocas
        )
        #disminuir la tasa de aprendizaje si después de 100 iteraciones la función de pérdida no ha disminuido
        planificador.step(acc)
        #guardar registros necesarios
        if acc>max_acc:
            #guardar las mejores firmas en un archivo aparte
            torch.save(image_syn.detach(),f"{ruta}Mejor.pt")
            max_acc=acc
            tqdm.write("Mejor accuracy")
            estancamiento=0
        else:
            estancamiento+=1
        acc_list.append(acc)
        torch.save(acc_list,ruta+"acc_list.pt")
    hist_perdida.append(perdida.item())
    torch.save(hist_perdida,ruta+"hist_perdida.pt")
    torch.save(image_syn,ruta+"image_syn.pt")
    torch.save(optimizer_img,ruta+"optimizer_img.pt")
    if parser.parse_args().epocas==0:
        ciclo.set_postfix(**{"pérdida":perdida.item()})
    else:
        tqdm.write(f"iteración {iteracion} pérdida destilado {perdida.item()} acc:{acc}")