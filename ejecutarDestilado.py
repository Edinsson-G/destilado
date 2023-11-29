import os
import matplotlib.pyplot as plt
import torch
from torchinfo import summary
import copy
from models import get_model
from perdida import *
from datos import *
from entrenamiento import train
import argparse
import numpy as np
from datasets import HyperX
from torch.utils.data import TensorDataset, DataLoader
from utils import embebido
#configurar argumentos
parser=argparse.ArgumentParser(description="Destilar imagenes hiperespectrales")
parser.add_argument("--modelo",
                    type=str,
                    choices=["nn","hamida","lee","chen","li"],
                    help="Nombre del modelo de red neuronal a utilizar en el destilado, es obligatorio si se quiere iniciar un nuevo destilado o si en la carpeta desde la cual se quieren cargar los datos no se encuntra un modelo, en tales casos debe ser de tipo cadena y alguna de las siguientes opciones: nn, hamida, lee, chen o li. Para mas información acerca de los modelos consulta la documentación")
parser.add_argument("--conjunto",
                    type=str,
                    choices=["PaviaC","PaviaU","IndianPines","KSC","Botswana"],
                    help="Nombre del conjunto de datos a destilar, es un argumento obligatorio si se va iniciar un nuevo destilado, debe ser de tipo cadena y una de las siguientes opciones: PaviaC, PaviaU, IndianPines, KSC o Botswana.")
parser.add_argument("--dispositivo",
                    type=int,
                    default=-1,
                    help="Indice del dispositivo en el que se ejecutará el algoritmo, si es negativo se ejecutará en la CPU. Su valor por defecto es -1.")
parser.add_argument("--semilla",type=int,help="Semilla pseudoaleatorio a usar.",default=0)
parser.add_argument("--historial",
                    type=bool,
                    default=True,
                    help="Si es verdadero se almacenará el historial de cambios de las firmas destiladas a lo largo de las iteraciones.")
parser.add_argument("--carpetaAnterior",
                    type=str,
                    help="Permite reanudar una ejecución anterior, debe ser de tipo cadena y contener la ruta en la cual se guardaron los archivos de dicha ejecucion. En caso de no querer cargar una ejecución anterior se deberá especificar el nombre de un modelo y conjunto de datos.")
parser.add_argument("--ipc",
                    type=int,
                    help="Indices por clase, obligatorio para iniciar un destilado nuevo")
parser.add_argument("--lrImg",
                    type=float,
                    default=0.01,
                    help="Tasa de aprendizaje del algoritmo de destilamiento, requerido solo en caso de iniciar un nuevo destilado.")
parser.add_argument("--iteraciones",
                    type=int,
                    help="Cantidad total de iteraciones a realizar.",
                    default=500)
parser.add_argument(
    "--inicializacion",
    type=str,
    choices=["muestreo","aleatoriedad"],
    help="Inicialización de las imágenes destiladas, muestreo significa seleccionar aleatoriamente muestras del conjunto de entrenamiento original y aleatoriedad significa inicializarlas siguiendo una distribución uniforme con números entre 0 y 1.",
    default="aleatoriedad"
)
device=torch.device("cpu" if parser.parse_args().dispositivo<0 else "cuda:"+str(parser.parse_args().dispositivo))
print("Algoritmo ejecutandose en",device)
#crear carpeta en la que se guardarán los resultados
carpeta="resultados"
if carpeta not in os.listdir('.'):
    os.mkdir(carpeta)
#lista con el nombre de algunas varibales a guardar o cargar segun sea el caso
para_guardar=["test_loader","val_loader","hiperparametros","optimizer_img","images_all","labels_all","label_syn","indices_class"]
if parser.parse_args().carpetaAnterior==None:#si se va iniciar un destilado nuevo
    parser.add_argument("--carpetaDestino",
                    type=str,
                    default="Modelo "+parser.parse_args().modelo+" conjunto "+parser.parse_args().conjunto+" ipc "+str(parser.parse_args().ipc)+" ritmo de aprendizaje "+str(parser.parse_args().lrImg),
                    help="Nombre de la subcarpeta en la que se almacenarán los resultados del algoritmo, debe estar en el directorio /resultados, si no existe entoces se creará.")
    #obtener modelos, optimizadores y datos
    ruta=carpeta+'/'+parser.parse_args().carpetaDestino+'/'
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
    train_loader=DataLoader(copy.deepcopy(dst_train),
                            batch_size=hiperparametros["batch_size"],
                            shuffle=False)
    test_loader=DataLoader(HyperX(img,test_gt,**hiperparametros),
                           batch_size=hiperparametros["batch_size"],
                           shuffle=False)
    val_loader= DataLoader(HyperX(img, val_gt, **hiperparametros),
                           batch_size=hiperparametros["batch_size"],
                           shuffle=False)
    del test_gt,val_gt,train_gt
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
    if parser.parse_args().carpetaDestino not in os.listdir(carpeta):
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
    parser.add_argument("--carpetaDestino",
                    type=str,
                    default=parser.parse_args().carpetaAnterior,
                    help="Nombre de la subcarpeta en la que se almacenarán los resultados del algoritmo, debe estar en el directorio /resultados, si no existe entoces se creará.")
    #obtener modelos, optimizadores y datos
    ruta=carpeta+'/'+parser.parse_args().carpetaDestino+'/'
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
        img_real=get_images(clase,hiperparametros["batch_size"],indices_class,images_all).to(device)
        img_sin=image_syn[clase*ipc:(clase+1)*ipc].reshape(tam)
        salida_real=embebido(net,img_real).detach()
        output_sin=embebido(net,img_sin)
        perdida+=torch.sum((torch.mean(salida_real,dim=0)-torch.mean(output_sin,dim=0))**2)
    optimizer_img.zero_grad()
    perdida.backward()
    print(image_syn.grad)
    optimizer_img.step()
    perdida_media+=perdida.item()
    perdida_media/=(num_classes)
    #cada 50 iteraciones se hará un entrenamiento real para evaluar el accuracy
    """
    if iteracion%50==0 or iteracion==parser.parse_args().iteraciones:
        for parametros in list(net.parameters()):
            parametros.requires_grad = True
        acc_entr,acc_test=train(net,
                                optimizador_red,
                                criterion,
                                train_loader,
                                400,
                                test_loader=val_loader,
                                device=device)
        hist_acc_train.append(acc_entr)
        hist_acc_val.append(acc_test)
        torch.save(hist_acc_train,ruta+"accuracyEntrenamiento.pt")
        torch.save(hist_acc_val,ruta+"accuracyTesteo.pt")
    """
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
    #cada 10 iteraciones se imprimirá el loss
    print("Iteración",iteracion,"perdida",perdida_media)
########################################################################################
"""
for it in range(ultima_iteracion,parser.parse_args().iteraciones+1):
    ''' Train synthetic data '''
    print("Iteración",it)
    net.train()
    net_parameters = list(net.parameters())
    optimizer_net.zero_grad()
    loss_avg = 0
    acc_train_acum=0
    acc_test_acum=0
    for ol in range(ol_inic,outer_loop):
        print(f'outer_loop ={ol}/{outer_loop}, iteration = {it}/{parser.parse_args().iteraciones}')
        ''' update synthetic data '''
        loss = torch.tensor(0.0).to(device)
        for c in range(num_classes):
            img_real = get_images(c, hiperparametros["batch_size"],indices_class,images_all).to(device)
            #img_real = get_images(c, batch_real,indices_class,images_all).to(device)
            lab_real = torch.ones((img_real.shape[0],), device=device, dtype=torch.long) * c
            net=net.to(device)
            output_real = net(img_real)
            loss_real = criterion(output_real, lab_real)
            gw_real = torch.autograd.grad(loss_real, net_parameters)
            gw_real = list((_.detach().clone() for _ in gw_real))
            img_syn = image_syn[c*ipc:(c+1)*ipc].reshape(tam)
            lab_syn = torch.ones((ipc,), device=device, dtype=torch.long) * c
            output_syn = net(img_syn)
            loss_syn = criterion(output_syn, lab_syn)
            gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

            loss += match_loss(gw_syn, gw_real,device)

        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()
        loss_avg += loss.item()
        if ol == outer_loop - 1:
            ol_inic=0
            break


        ''' update network '''
        image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
        dst_syn_train =TensorDataset(image_syn_train,label_syn_train)
        trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=hiperparametros["batch_size"], shuffle=False, num_workers=0)
        #trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=batch_train, shuffle=False, num_workers=0)
        acc_train,acc_test=train(net,
                optimizer_net,
                criterion,
                trainloader,
                inner_loop,
                test_loader=test_loader,
                device=device,
                )
        print("     Accuracy de entrenamiento:",acc_train)
        print("     Accuracy de validación:",acc_test)
        acc_train_acum=acc_train_acum+acc_train
        acc_test_acum=acc_test_acum+acc_test
        #guardar optimizador
        torch.save(optimizer_img,ruta+"optimizer_img")
        torch.save(optimizer_net,ruta+"optimizer_net")
        #guardar firmas sinteticas o historial de firmas sinteticas si corresponde
        if parser.parse_args().historial:
          historial_imagenes_sinteticas.append(copy.deepcopy(image_syn_train))
          #se guarda el historial en un archivo de nombre img
          torch.save(historial_imagenes_sinteticas,ruta+"imgs")
        else:
          #guardar unicamente las firmas en la iteración actual
          torch.save(image_syn_train,ruta+"imgs")
        torch.save(net,ruta+"net")
        torch.save(ol,ruta+"ol")
    net.eval()
    perdida_destilado.append(loss_avg/outer_loop)
    #guardar el loss en un archivo llamado perdida
    torch.save(perdida_destilado,ruta+"perdida")
    #guardar accuracies
    hist_acc_train.append(acc_train_acum/outer_loop)
    hist_acc_test.append(acc_test_acum/outer_loop)
    torch.save(hist_acc_train,ruta+"accuracyEntrenamiento")
    torch.save(hist_acc_test,ruta+"accuracyTesteo")
    print("iteracion:",it,
          "perdida promedio:",perdida_destilado[-1],
          "accuracy promedio de entrenamiento:",hist_acc_train[-1],
          "accuracy promedio de validación:",hist_acc_test[-1])

    ultima_iteracion=ultima_iteracion+1
    net,optimizer_net,temp1,temp2=get_model(hiperparametros["model"],device, **hiperparametros)
    del temp1,temp2
    torch.save(net,ruta+"net")
    torch.save(optimizer_net,ruta+"optimizer_net")
"""