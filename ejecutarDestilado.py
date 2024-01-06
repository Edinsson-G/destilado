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
from torch.utils.data import DataLoader,TensorDataset
from utils import embebido
import warnings
from tqdm import tqdm
from entrenamiento import train
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
                    default=0.001,
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
parser.add_argument(
    "--reanudar",
    type=bool,
    default=False,
    help="En caso que no se haya especificado una carpeta anterior verificar si la carpeta destino existe para reanudar el destilado que contiene."
)
parser.add_argument(
    "--cicloInterno",
    type=int,
    default=2,
    help="Cantidad de epocas a entrenar la red por cada ciclo externo"
)
parser.add_argument(
    "--cicloExterno",
    default=1,
    type=int,
    help="Cantidad de ciclos internos por cada iteración"
)
parser.add_argument(
    "--aumentoVal",
    default=False,
    type=bool,
    help="Si es True y tecAumento es diferente de None se hará un aumento similar al de cada destilado al momento de validar el rendimiento del modelo."
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
#lista de variables que se deberán guardar o carguar segun sea el caso
#variables que solo se deben carguar o guardar una sola vez durante la ejecucion
global_var=["test_loader","val_loader","images_all","labels_all","label_syn","indices_class","hiperparametros","train_loader"]
#variables que se guardarán cada ol
ol_var=[
    "ult_ol",#ultimo ol completado de la epoca actual
    "perd_acum",#perdida acumulada en el ol
    "image_syn"
]
#variables que se guardarán al finalizar cada epoca
ep_var=["hist_perdida","hist_acc"]
#definir carpetas anteriores y destino
#destino=f"ritmo+de+aprendizaje+{parser.parse_args().lrImg}+aumento+{parser.parse_args().tecAumento}+ol+{parser.parse_args().cicloExterno}+innLoop+{parser.parse_args().cicloInterno}"if parser.parse_args().carpetaDestino==None else parser.parse_args().carpetaDestino
destino=f"{parser.parse_args().tecAumento}_{parser.parse_args().aumentoVal}_{parser.parse_args().factAumento}" if parser.parse_args().carpetaDestino==None else parser.parse_args().carpetaDestino
carpetaAnterior=parser.parse_args().carpetaAnterior
if carpetaAnterior==None and parser.parse_args().reanudar and destino in os.listdir(carpeta):
    carpetaAnterior=destino
if destino not in os.listdir(carpeta):
    os.mkdir(carpeta+'/'+destino)
if carpetaAnterior!=None:
    #verificar la existencia de todos los archivos necesarios para reanudar el destilado (para ello se debió haber realizado por lo menos un ol)
    reanudar=True
    for nombre_archivo in ol_var+global_var:
        if nombre_archivo+".pt" not in os.listdir(carpeta+'/'+destino):
            reanudar=False
            warnings.warn(f"No se encontó el archivo {carpeta}/{destino}/{nombre_archivo}.pt, se iniciará el entrenamiento desde 0.")
    if not reanudar:
        carpetaAnterior=None
ruta=carpeta+'/'+destino+'/'
if carpetaAnterior==None:#si se va iniciar un destilado nuevo
    print("Iniciando nuevo destilado en",ruta)
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
    summary(net)
    train_gt,test_gt=sample_gt(gt,
                               hiperparametros["training_sample"],
                               mode=hiperparametros["sampling_mode"])
    train_gt, val_gt = sample_gt(train_gt, 0.8, mode="random")
    dst_train = HyperX(img, train_gt, **hiperparametros)
    train_loader=DataLoader(dst_train,batch_size=hiperparametros["batch_size"],shuffle=True)
    dst_test=HyperX(img,test_gt,**hiperparametros)
    test_loader=DataLoader(dst_test,batch_size=len(dst_test),shuffle=True)
    dst_val=HyperX(img, val_gt, **hiperparametros)
    val_loader= DataLoader(dst_val,batch_size=len(dst_val),shuffle=True)
    del test_gt,val_gt,train_gt,dst_val,dst_test
    channel=img.shape[-1]
    clases=np.unique(gt)
    num_classes=clases.size
    for etiqueta_ingnorada in hiperparametros["ignored_labels"]:
        if etiqueta_ingnorada in clases:
            num_classes=num_classes-1
    del img,gt,clases
    #hiperparametros["n_classes"]=num_classes
    #preprocesar datos reales
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))] # Save the images (1,1,28,28)
    labels_all = [int(dst_train[i][1]) for i in range(len(dst_train))] # Save the labels
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
    tam=list(images_all.shape)
    tam[0]=num_classes*ipc
    if parser.parse_args().inicializacion=="muestreo":
        image_syn=torch.empty(tam,device=device)
        import secrets
        for i,clase in enumerate(label_syn):
            image_syn[i]=images_all[secrets.choice(indices_class[clase])]
        image_syn=image_syn.clone().detach().requires_grad_(True)
    else:
        image_syn=torch.rand(tam,requires_grad=True,device=device)
    del tam
    optimizer_img = torch.optim.SGD([image_syn], lr=parser.parse_args().lrImg, momentum=0.5)
    #variables necesarias para reanudar destilado
    if parser.parse_args().historial:
        historial_imagenes_sinteticas=[]
    hist_perdida=[]
    hist_acc=[]
    ult_ol=0
    #guardar variables globales
    for variable,archivo in zip(
        [
            test_loader,
            val_loader,
            images_all,
            labels_all,
            label_syn,
            indices_class,
            hiperparametros,
            train_loader
        ],
        global_var
    ):
        torch.save(variable,ruta+archivo+".pt")
    #durante el destilado no se utilizará esta variable
    del test_loader
else:#se va a reanudar un entrenatiento previo
    print("Reanudando entrenamiento en",ruta)
    #obtener modelos, optimizadores y datos
    del global_var[0]#no es necesario cargar test_loader
    ruta_anterior=carpeta+'/'+carpetaAnterior+'/'
    #restablecer el estado del generador de números pseudoaleatorios
    torch.set_rng_state(torch.load(ruta_anterior+"tensorSemilla.pt"))
    #carguar las variables de global_var y ol_var
    (val_loader,images_all,labels_all,label_syn,indices_class,hiperparametros,train_loader,ult_ol,perd_acum,image_syn)=tuple(torch.load(ruta+archivo+".pt",map_location=device)for archivo in global_var+ol_var)
    #carguar las variables de ep_var
    hist_perdida=torch.load(ruta+"hist_perdida.pt")if os.path.exists(ruta+"hist_perdida.pt")else []
    hist_acc=torch.load(ruta+"hist_acc.pt")if os.path.exists(ruta+"hist_acc.pt")else []
    if type(image_syn)==list:
        #en el entrenamiento anterior el argumento --model fue True, image_syn es realemente el último elemento de ese historial
        historial_imagenes_sinteticas=copy.deepcopy(image_syn)
        image_syn=image_syn[-1]
    optimizer_img = torch.optim.SGD([image_syn], lr=parser.parse_args().lrImg, momentum=0.5)
    #modelo y optimizadores
    net,optimizador_red,criterion,_= get_model(hiperparametros["model"],
                                               hiperparametros["device"],**hiperparametros)
    num_classes=hiperparametros["n_classes"]-1
    ipc=int(len(label_syn)/hiperparametros["n_classes"])
    channel=image_syn.shape[1]
print("Los resultados se guardarán en ",ruta)
#optimizer_img for synthetic data
planificador=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_img,"max",patience=100,verbose=True)
for iteracion in range(len(hist_perdida),parser.parse_args().iteraciones+1):
    print("iteración",iteracion)
    perd_acum=0
    #actualizar imágenes sintéticas
    for ol in tqdm(range(ult_ol,parser.parse_args().cicloExterno)):
        net.train()
        perdida=torch.tensor(0.0).to(device)
        for parametros in list(net.parameters()):
            parametros.requires_grad = False
        for clase in range(num_classes):
            #salida sin aumento
            img_real=get_images(clase,hiperparametros["batch_size"],indices_class,images_all).to(device)
            img_sin=image_syn[clase*ipc:(clase+1)*ipc]
            #aplicar aumento
            """
            if parser.parse_args().tecAumento=="ruido":
                img_real=adicion(img_real,parser.parse_args().factAumento)
                img_sin=adicion(img_sin,parser.parse_args().factAumento)
            elif parser.parse_args().tecAumento!="None":
                if parser.parse_args().tecAumento=="escalamiento":
                    #paramAumento=torch.clip(torch.rand(parser.parse_args().factAumento),0.01,0.99)
                    #ruido entre 0.85 y 1.1
                    paramAumento=torch.rand(parser.parse_args().factAumento)/4+0.85
                else:
                    #tecnica de aumento potencia
                    paramAumento=torch.rand(parser.parse_args().factAumento)/20
                img_real=noAdicion(img_real,paramAumento,parser.parse_args().tecAumento)
                img_sin=noAdicion(img_sin,paramAumento,parser.parse_args().tecAumento)
            """
            if parser.parse_args().tecAumento!="None":
                img_real=aumento(parser.parse_args().tecAumento,parser.parse_args().factAumento,img_real)
                img_sin=aumento(parser.parse_args().tecAumento,parser.parse_args().factAumento,img_sin)
            #aplicar embebido
            salida_real=embebido(net,img_real).detach()
            output_sin=embebido(net,img_sin)
            #funcion de perdida
            perdida+=torch.sum((torch.mean(salida_real,dim=0)-torch.mean(output_sin,dim=0))**2)+torch.sum((torch.std(salida_real,dim=0)-torch.std(output_sin,dim=0))**2)
        optimizer_img.zero_grad()
        perdida.backward()
        optimizer_img.step()
        #entrenar red
        for parametros in list(net.parameters()):
            parametros.requires_grad = True
        (net,)=train(
            net,
            optimizador_red,
            criterion,
            train_loader,
            parser.parse_args().cicloInterno,
            device=device
        )
        #guardar generador de números pseuadoaleatorios
        torch.save(torch.get_rng_state(),ruta+"tensorSemilla.pt")
        #guardar registros de ol
        perd_acum+=perdida.item()
        ult_ol=ol
        torch.save(ult_ol,ruta+"ult_ol.pt")
        torch.save(perd_acum,ruta+"perd_acum.pt")
        if parser.parse_args().historial:
            historial_imagenes_sinteticas.append(copy.deepcopy(image_syn).to("cpu"))
            torch.save(historial_imagenes_sinteticas,ruta+"image_syn.pt")
        else:
            torch.save(image_syn,ruta+"image_syn.pt")
    #validar red
    #reinicar red
    net,optimizador_red,criterion,_= get_model(hiperparametros["model"],hiperparametros["device"],**hiperparametros)
    #aumentar datos si es el caso
    if parser.parse_args().aumentoVal:
        entr_val,etq_val=aumento(
            parser.parse_args().tecAumento,
            parser.parse_args().factAumento,
            copy.deepcopy(image_syn).detach(),
            label_syn
        )
    else:
        entr_val=copy.deepcopy(image_syn).detach()
        etq_val=label_syn
    _,acc=train(
        net,
        optimizador_red,
        criterion,
        DataLoader(
            TensorDataset(entr_val,etq_val),
            batch_size=hiperparametros["batch_size"],
            shuffle=True,num_workers=0
        ),
        test_loader=val_loader,#se usarán los datos de validación cómo si fueran los de testeo en este caso
        device=device
    )
    #reiniciar pesos de la red para la siguiente iteración
    net,optimizador_red,criterion,_= get_model(hiperparametros["model"],hiperparametros["device"],**hiperparametros)
    #disminuir la tasa de aprendizaje si después de 100 iteraciones la función de pérdida no ha disminuido
    planificador.step(acc)
    #guardar registros necesarios
    hist_acc.append(acc)
    hist_perdida.append(perd_acum/10)
    for variable,archivo in zip((hist_perdida,hist_acc),ep_var):
        torch.save(variable,ruta+archivo+".pt")
    print("iteracion:",iteracion,"pérdida:",hist_perdida[-1],"accuracy:",acc)