import os
import torch
from torchinfo import summary
import copy
from models import get_model
from perdida import *
from datos import *
import argparse
import numpy as np
from datasets import HyperX
from torch.utils.data import DataLoader
from utils import embebido
import warnings
#configurar argumentos
parser=argparse.ArgumentParser(description="Destilar imagenes hiperespectrales")
parser.add_argument("--modelo",
                    type=str,
                    choices=["nn","hamida","lee","chen","li"],
                    default="nn",
                    help="Nombre del modelo de red neuronal a utilizar en el destilado, es obligatorio si se quiere iniciar un nuevo destilado.")
parser.add_argument("--conjunto",
                    type=str,
                    choices=["PaviaC","PaviaU","IndianPines","KSC","Botswana"],
                    default="IndianPines",
                    help="Nombre del conjunto de datos a destilar, obligatorio si se va a realizar un destilado nuevo.")
parser.add_argument("--dispositivo",
                    type=int,
                    default=-1,
                    help="Indice del dispositivo en el que se ejecutará el algoritmo, si es negativo se ejecutará en la CPU.")
parser.add_argument("--semilla",type=int,help="Semilla pseudoaleatorio a usar.",default=0)
parser.add_argument("--historial",
                    type=bool,
                    default=True,
                    help="Si es verdadero se almacenará el historial de cambios de las firmas destiladas a lo largo de las iteraciones.")
parser.add_argument("--carpetaAnterior",
                    type=str,
                    help="Para reanudar un destilado anterior, debe especificar el nombre de la carpeta que contiene los registros de dicho destilado.")
parser.add_argument("--ipc",
                    type=int,
                    default=10,
                    help="Imagenes por clase, obligatorio para iniciar un destilado nuevo")
parser.add_argument("--lrImg",
                    type=float,
                    default=0.01,
                    help="Tasa de aprendizaje del algoritmo de destilamiento, requerido solo en caso de iniciar un nuevo destilado.")
parser.add_argument("--iteraciones",
                    type=int,
                    help="Cantidad total de iteraciones a realizar.",
                    default=500)
parser.add_argument(
    "--factAumento",
    type=int,
    default=2,
    help="Factor de aumento, cantidad de muestras nuevas a generar por cada ejemplo en el aumento de datos."
)
parser.add_argument(
    "--tecAumento",
    type=str,
    choices=["ruido","escalamiento","potencia","None"],
    default="ruido",
    help="Tecnica con la que se hará el aumento de datos, obligatoria si fracAumento es positiva"
)
parser.add_argument(
    "--inicializacion",
    type=str,
    choices=["muestreo","aleatoriedad"],
    help="Inicialización de las imágenes destiladas, muestreo significa seleccionar aleatoriamente muestras del conjunto de entrenamiento original y aleatoriedad significa inicializarlas siguiendo una distribución uniforme con números entre 0 y 1.",
    default="aleatoriedad"
)
parser.add_argument(
    "--carpetaDestino",
    type=str,
    help="Nombre de la subcarpeta en la que se almacenarán los resultados del algoritmo, debe estar en el directorio /resultados, si no existe entoces se creará."
)
device=torch.device("cpu" if parser.parse_args().dispositivo<0 else "cuda:"+str(parser.parse_args().dispositivo))
torch.set_default_device(device)
if (parser.parse_args().factAumento>0) and (parser.parse_args().tecAumento==None):
    exit("Se especificó un factor de aumento de aumento pero no un método de aumento.")
elif parser.parse_args().factAumento<0:
    exit("El factor de aumento no debe ser negativo.")
elif (parser.parse_args().tecAumento=="ruido") and (parser.parse_args().factAumento==0):
    warnings.warn("Se especificó la técnica de aumento de ruido; sin embargo el factor de aumento es 0, por lo que no se realizará ningún aumento.")
print("Algoritmo ejecutandose en",device)
#crear carpeta en la que se guardarán los resultados
carpeta="resultados"
if carpeta not in os.listdir('.'):
    os.mkdir(carpeta)
#lista con el nombre de algunas varibales a guardar o cargar segun sea el caso
para_guardar=["test_loader","val_loader","hiperparametros","optimizer_img","images_all","labels_all","label_syn","indices_class"]
if parser.parse_args().carpetaAnterior==None:#si se va iniciar un destilado nuevo
    if parser.parse_args().carpetaDestino!=None:
        ruta=carpeta+'/'+parser.parse_args().carpetaDestino+'/'
    else:
        ruta=f"{carpeta}/Modelo {parser.parse_args().modelo} conjunto {parser.parse_args().conjunto} ipc {parser.parse_args().ipc} ritmo de aprendizaje {parser.parse_args().lrImg} aumento {parser.parse_args().tecAumento}/"
    #obtener modelos, optimizadores y datos
    torch.manual_seed(parser.parse_args().semilla)
    #carguar imagenes
    img,gt,_,IGNORED_LABELS,_,_= get_dataset(parser.parse_args().conjunto,"Datasets/")
    gt=np.array(gt,dtype=np.int32)
    hiperparametros={'dataset':parser.parse_args().conjunto,
                'model':parser.parse_args().modelo,
                'folder':'./Datasets/',
                'cuda':parser.parse_args().dispositivo,
                'runs': 1,
                'training_sample': 0.8,
                'sampling_mode': 'random',
                'class_balancing': False,
                'test_stride': 1,
                'flip_augmentation': False,
                'radiation_augmentation': False,
                'mixture_augmentation': False,
                'with_exploration': False,
                'n_classes':np.unique(gt).size,
                'n_bands':img.shape[-1],
                'ignored_labels':IGNORED_LABELS,
                'device': device}
    #redefinir las etiquetas entre 0 y num_clases puesto que se ignorará la etiqueta 0
    if 0 in hiperparametros["ignored_labels"]:
      gt=gt-1
      hiperparametros["ignored_labels"]=(
          torch.tensor(hiperparametros["ignored_labels"])-1
          ).tolist()
    net,optimizador_red,criterion,hiperparametros= get_model(hiperparametros["model"],
                                                             hiperparametros["device"],
                                                             **hiperparametros)
    train_gt,test_gt=sample_gt(gt,
                               hiperparametros["training_sample"],
                               mode=hiperparametros["sampling_mode"])
    train_gt, val_gt = sample_gt(train_gt, 0.8, mode="random")
    dst_train = HyperX(img, train_gt, **hiperparametros)
    dst_test=HyperX(img,test_gt,**hiperparametros)
    test_loader=DataLoader(dst_test,
                           batch_size=len(dst_test),
                           shuffle=True)
    dst_val=HyperX(img, val_gt, **hiperparametros)
    val_loader= DataLoader(dst_val,
                           batch_size=len(dst_val),
                           shuffle=True)
    del test_gt,val_gt,train_gt,dst_val,dst_test
    channel=img.shape[-1]
    clases=np.unique(gt)
    num_classes=clases.size
    for etiqueta_ingnorada in hiperparametros["ignored_labels"]:
        if etiqueta_ingnorada in clases:
            num_classes=num_classes-1
    del img,gt,clases
    hiperparametros["n_classes"]=num_classes
    ultima_iteracion=0
    ol_inic=0
    #preprocesar datos reales
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))] # Save the images (1,1,28,28)
    labels_all = [int(dst_train[i][1]) for i in range(len(dst_train))] # Save the labels
    del dst_train
    for i, lab in enumerate(labels_all): # Save the index of each class labels
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to(device) # Cat images along the batch dimension
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=device) # Make the labels a tensor
    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c]))) # Prints how many labels are for each class
    ipc=parser.parse_args().ipc
    #etiquetas sintéticas de la forma [0,0,1,1,...,num_classes-1] cada etiqueta repitiendose ipc veces.
    label_syn=torch.repeat_interleave(torch.arange(num_classes,requires_grad=False),ipc)
    #Inicialización de imagenes sintéticas
    tam=(num_classes*ipc,channel)
    if hiperparametros["patch_size"]>1:
        tam=tam+(hiperparametros["patch_size"],hiperparametros["patch_size"])
    if parser.parse_args().inicializacion=="muestreo":
        image_syn=torch.empty(tam,device=device)
        import secrets
        for i,clase in enumerate(label_syn):
            image_syn[i]=images_all[secrets.choice(indices_class[clase])]
        image_syn=image_syn.clone().detach().requires_grad_(True)
    else:
        image_syn=torch.rand(tam,requires_grad=True,device=device)
    del tam
    optimizer_img = torch.optim.SGD([image_syn], lr=parser.parse_args().lrImg, momentum=0.5) # optimizer_img for synthetic data
    hist_perdida=[]
    summary(net)
    hist_acc_train=[]
    hist_acc_val=[]
    #crear la carpeta si no existe
    if not os.path.isdir(rf"{ruta}"):
        os.mkdir(ruta)
    #guardar archivos necesarios para reanudar el entrenamiento
    for variable,archivo in zip(
        [
            test_loader,
            val_loader,
            hiperparametros,
            optimizer_img,
            images_all,
            labels_all,
            label_syn,
            indices_class
        ],
        para_guardar
    ):
        torch.save(variable,ruta+archivo+".pt")
    del test_loader
    if parser.parse_args().historial:
        historial_imagenes_sinteticas=[copy.deepcopy(image_syn)]
        torch.save(historial_imagenes_sinteticas,ruta+"imgs.pt")
    else:
        torch.save(image_syn,ruta+"imgs.pt")
else:#se va a reanudar un entrenatiento previo
    if parser.parse_args().carpetaDestino!=None:
        ruta=carpeta+'/'+parser.parse_args().carpetaDestino+'/'
    else:
        #por defecto sobreeescribir en la carpeta anterior
        ruta=carpeta+'/'+parser.parse_args().carpetaAnterior+'/'
    #obtener modelos, optimizadores y datos
    del para_guardar[0]#no es necesario cargar test_loader
    ruta_anterior=carpeta+'/'+parser.parse_args().carpetaAnterior+'/'
    #restablecer el estado del generador de números pseudoaleatorios
    torch.set_rng_state(torch.load(ruta_anterior+"tensorSemilla.pt"))
    #carguar las variables de la lista para_guardar
    (val_loader,hiperparametros,optimizer_img,images_all,labels_all,label_syn,indices_class,hist_perdida,hist_acc_val,hist_acc_train,image_syn)=tuple(torch.load(ruta+archivo+".pt")for archivo in para_guardar+["histPerdida","accEnt","accval","imgs"])
    if type(image_syn)==list:
        #en el entrenamiento anterior el argumento --model fue True, image_syn es realemente el último elemento de ese historial
        historial_imagenes_sinteticas=copy.deepcopy(image_syn)
        image_syn=image_syn[-1]
    #modelo y optimizadores
    net,optimizador_red,criterion,_= get_model(hiperparametros["model"],
                                               hiperparametros["device"],**hiperparametros)
    num_classes=hiperparametros["n_classes"]
    ipc=int(len(label_syn)/hiperparametros["n_classes"])
    channel=image_syn.shape[1]
print("Los resultados se guardarán en ",ruta)
tam=(ipc, channel)
if hiperparametros["model"]=="hamida":
    tam=(ipc,
         1,
         channel,
         hiperparametros["patch_size"],
         hiperparametros["patch_size"])
elif hiperparametros["patch_size"]>1:
    tam=tam+(hiperparametros["patch_size"],hiperparametros["patch_size"])
for iteracion in range(len(hist_perdida),parser.parse_args().iteraciones+1):
    net.train()
    for parametros in list(net.parameters()):
        parametros.requires_grad = False
    perdida_media=0
    #actualizar imágenes sintéticas
    perdida=torch.tensor(0.0).to(device)
    for clase in range(num_classes):
        #salida sin aumento
        img_real=get_images(clase,hiperparametros["batch_size"],indices_class,images_all).to(device)
        img_sin=image_syn[clase*ipc:(clase+1)*ipc].reshape(tam)
        #aplicar aumento
        if parser.parse_args().tecAumento=="ruido":
            img_real=adicion(img_real,parser.parse_args().factAumento)
            img_sin=adicion(img_sin,parser.parse_args().factAumento)
        elif parser.parse_args().tecAumento=="escalamiento"or parser.parse_args().tecAumento=="potencia":
            paramAumento=torch.clip(torch.rand(parser.parse_args().factAumento),0.01,0.99)
            img_real=noAdicion(img_real,paramAumento,parser.parse_args().tecAumento)
            img_sin=noAdicion(img_sin,paramAumento,parser.parse_args().tecAumento)
        #aplicar embebido
        salida_real=embebido(net,img_real).detach()
        output_sin=embebido(net,img_sin)
        #funcion de perdida
        perdida+=torch.sum((torch.mean(salida_real,dim=0)-torch.mean(output_sin,dim=0))**2)
    optimizer_img.zero_grad()
    perdida.backward()
    optimizer_img.step()
    perdida_media+=perdida.item()
    perdida_media/=(num_classes)
    #reinicializar los pesos de la red
    net,optimizador_red,criterion,hiperparametros= get_model(hiperparametros["model"],
                                                             hiperparametros["device"],
                                                             **hiperparametros)
    #guardar registros necesarios
    hist_perdida.append(perdida_media)
    if parser.parse_args().historial:
        historial_imagenes_sinteticas.append(copy.deepcopy(image_syn))
        torch.save(historial_imagenes_sinteticas,ruta+"imgs.pt")
    else:
        torch.save(image_syn,ruta+"imgs.pt")
    for variable,archivo in zip(
        [torch.get_rng_state(),hist_perdida,hist_acc_val,hist_acc_train],
        ["tensorSemilla","histPerdida","accEnt","accval"]
    ):
        torch.save(variable,ruta+archivo+".pt")
    print("Iteración",iteracion,"perdida",perdida_media)