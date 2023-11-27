import os
import torch.nn as nn
import torch
import secrets
from torch.utils.data import TensorDataset
from torchinfo import summary
import copy
import time
from models import get_model
from perdida import *
from datos import *
from entrenamiento import train
import argparse
#configurar argumentos
parser=argparse.ArgumentParser(description="Destilar imagenes hiperespectrales")
parser.add_argument("--modelo",
                    type=str,
                    choices=["nn","hamida","lee","chen","li"],
                    help="Nombre del modelo de re neuronal a utilizar en el destilado, es obligatorio si se quiere iniciar un nuevo destilado o si en la carpeta desde la cual se quieren cargar los datos no se encuntra un modelo, en tales casos debe ser de tipo cadena y alguna de las siguientes opciones: nn, hamida, lee, chen o li. Para mas información acerca de los modelos consulta la documentación")
parser.add_argument("--dataset",
                    type=str,
                    choices=["PaviaC","PaviaU","IndianPines","KSC","Botswana"],
                    help="Nombre del conjunto de datos a destilar, es un argumento obligatorio si se va iniciar un nuevo destilado, debe ser de tipo cadena y una de las siguientes opciones: PaviaC, PaviaU, IndianPines, KSC o Botswana.")
parser.add_argument("--dispositivo",
                    type=int,
                    default=-1,
                    help="Indice del dispositivo en el que se ejecutará el algoritmo, si es negativo se ejecutará en la CPU. Su valor por defecto es -1.")
parser.add_argument("--semilla",type=int,help="Semilla pseudoaleatorio a usar (opcional).")
parser.add_argument("--historial",
                    type=bool,
                    default=False,
                    help="Si es verdadero se almacenará el historial de cambios de las firmas destiladas a lo largo de las iteraciones, por defecto es falso.")
parser.add_argument("--carpetaAnterior",
                    type=str,
                    help="Permite reanudar una ejecución anterior, debe ser de tipo cadena y contener la ruta en la cual se guardaron los archivos de dicha ejecucion. En caso de no querer cargar una ejecución anterior se deberá especificar el nombre de un modelo y conjunto de datos.")
parser.add_argument("--carpetaDestino",
                    type=str,
                    default=time.strftime("%c"),
                    help="Nombre de la subcarpeta en la que se almacenarán los resultados del algoritmo, debe estar en el directorio /resultados, si no existe entoces se creará.")
parser.add_argument("--ipc",
                    type=int,
                    help="Indices por clase, obligatorio para iniciar un destilado nuevo")
parser.add_argument("--lrImg",
                    type=float,
                    help="Tasa de aprendizaje del algoritmo de destilamiento, requerido solo en caso de iniciar un nuevo destilado.")
parser.add_argument("--iteraciones",
                    type=int,
                    help="Cantidad total de iteraciones a realizar.",
                    required=True)
device=torch.device("cpu" if parser.parse_args().dispositivo<0 else "cuda:"+str(parser.parse_args().dispositivo))
print("Algoritmo ejecutandose en",device)
#crear carpeta en la que se guardarán los resultados
carpeta="resultados/"
#si se especificó una semilla entonces configurarla
if parser.parse_args().semilla!=None:
   torch.manual_seed(parser.parse_args().semilla)
#obtener modelos, optimizadores y datos
ruta=carpeta+parser.parse_args().carpetaDestino+'/'
#lista con el nombre de algunas varibales a guardar o cargar segun sea el caso
para_guardar=["dst_train",
              "dst_test",
              "dst_val",
              "train_loader",
              "test_loader",
              "val_loader",
              "hiperparametros",
              "criterion",
              "optimizer_net",
              "optimizer_img",
              "images_all",
              "labels_all",
              "label_syn",#"etiquetas"
              "net",
              "indices_class"]
if parser.parse_args().carpetaAnterior==None:#si se va iniciar un destilado nuevo
    (channel,
     num_classes,
     mean,
     std,
     dst_train,
     dst_test,
     dst_val,
     train_loader,
     test_loader,
     val_loader,
     net,
     optimizer_net,
     criterion,
     hiperparametros
    )=obtener_datos(parser.parse_args().dataset,
                          parser.parse_args().dispositivo,
                          parser.parse_args().modelo)
    ultima_iteracion=0
    ol_inic=0
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
    #incializar con muestras aleatorias de cada clase 
    #etiquetas sinteticas de la forma [0,0,1,1,1,...,num_classes,num_classes,num_classes] (la cantidad real de repeticiones las determina el ipc)
    ipc=parser.parse_args().ipc
    label_syn=torch.repeat_interleave(torch.arange(num_classes,requires_grad=False),ipc)
    #imagenes sintéticas
    tam=(num_classes*ipc,channel)
    if hiperparametros["patch_size"]>1:
        tam=tam+(hiperparametros["patch_size"],hiperparametros["patch_size"])
    image_syn=torch.empty(tam,device=device)
    del tam
    for i,clase in enumerate(label_syn):
        image_syn[i]=images_all[secrets.choice(indices_class[clase])]
    optimizer_img = torch.optim.SGD([image_syn], lr=parser.parse_args().lrImg) # optimizer_img for synthetic data
    #image_syn=torch.tensor(image_syn,requires_grad=True)
    image_syn=image_syn.clone().detach().requires_grad_(True)
    perdida_destilado=[]
    summary(net)
    hist_acc_train=[]
    hist_acc_test=[]
    #crear la carpeta si no existe
    if parser.parse_args().carpetaDestino not in os.listdir(carpeta):
        os.mkdir(ruta)
    #guardar archivos necesarios para reanudar el entrenamiento
    for nombre_variable in para_guardar:
        #globals()[nombre_variable] devuelve la variable de nombre "nombre_variable"
        torch.save(globals()[nombre_variable],ruta+nombre_variable)
    torch.save(parser.parse_args().historial,ruta+"guardar_historial")
    if parser.parse_args().historial:
        historial_imagenes_sinteticas=[copy.deepcopy(image_syn)]
        torch.save(historial_imagenes_sinteticas,ruta+"imgs")
    else:
        torch.save(image_syn,ruta+"imgs")
else:#se va a carguar un entrenatiento previo
    device=torch.device("cpu"if parser.parse_args().dispositivo<0 else "cuda:"+parser.parse_args().dispositivo)
    ruta_anterior=carpeta+parser.parse_args().carpetaAnterior+'/'
    for nombre_variable in para_guardar:
        #crea la variable nombre_variable y carga en ella el objeto correspondiente
        globals()[nombre_variable]=torch.load(ruta_anterior+nombre_variable,
                                              map_location=device)
    hiperparametros["cuda"]=parser.parse_args().dispositivo
    hiperparametros["device"]=device
    ipc=int(len(label_syn)/(hiperparametros["n_classes"]-1))
    hist_acc_train=torch.load(ruta_anterior+"accuracyEntrenamiento")
    hist_acc_test=torch.load(ruta_anterior+"accuracyTesteo")
    perdida_destilado=torch.load(ruta_anterior+"perdida")
    ol_inic=torch.load(ruta_anterior+"ol")
    ultima_iteracion=len(hist_acc_train)
    print(torch.tensor(hist_acc_train).shape)
    image_syn=torch.load(ruta_anterior+"imgs",map_location=device)
    if torch.load(ruta_anterior+"guardar_historial"):#el entrenamiento que estamos carguando guardó el historial de imagenes destiladas
        if parser.parse_args().historial:
            historial_imagenes_sinteticas=copy.deepcopy(image_syn)
        image_syn=image_syn[-1]
    elif parser.parse_args().historial:
            historial_imagenes_sinteticas=[copy.deepcopy(image_syn)]
    channel=image_syn.shape[1]
    num_classes=hiperparametros["n_classes"]-1
print("Los resultados se guardarán en ",ruta)
outer_loop, inner_loop = get_loops(ipc) # Get the two hyper-parameters of outer-loop and inner-loop
optimizer_img.zero_grad()
criterion = nn.CrossEntropyLoss().to(device) # Loss function
tam=(ipc, channel)
if hiperparametros["model"]=="hamida":
    tam=(ipc,
         1,
         channel,
         hiperparametros["patch_size"],
         hiperparametros["patch_size"])
elif hiperparametros["patch_size"]>1:
    tam=tam+(hiperparametros["patch_size"],hiperparametros["patch_size"])
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

            #loss += match_loss(gw_syn, gw_real,device)
            loss += match_loss_extend(gw_syn, gw_real,"mse")

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
    print("iteracion:",
          it,
          "perdida promedio:",
          perdida_destilado[-1],
          "accuracy promedio de entrenamiento:",
          hist_acc_train[-1],
          "accuracy promedio de validación:",
          hist_acc_test[-1])

    ultima_iteracion=ultima_iteracion+1
    net,optimizer_net,temp1,temp2=get_model(hiperparametros["model"],device, **hiperparametros)
    del temp1,temp2
    torch.save(net,ruta+"net")
    torch.save(optimizer_net,ruta+"optimizer_net")