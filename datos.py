import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset,TensorDataset, DataLoader
import matplotlib.pyplot as plt
from datasets import get_dataset,HyperX
from utils import sample_gt
from models import get_model
import math
from perdida import *
import os
import warnings
#definir hiperparametros a utilizar según el modelo
class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]
nombre_datos="IndianPines"
ruta="Datasets/"
#funcion para intercambiar 2 elementos de un vector
def intercambiar(vector,indice1,indice2):
  temporal=vector[indice1]
  vector[indice1]=vector[indice2]
  vector[indice2]=temporal
  return vector
def obtener_datos(conjunto,dispositivo,modelo):
    img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(conjunto, "Datasets/")
    gt=np.array(gt,dtype=np.int32)
    #hiperparametros que se usarán en este algoritmo
    hyperparams={'dataset':conjunto,
                'model': modelo,
                'folder':'./Datasets/',
                'cuda': dispositivo,
                'runs': 1,
                'training_sample': 0.8,
                'sampling_mode': 'random',
                'class_balancing': False,
                'test_stride': 1,
                'flip_augmentation': False,
                'radiation_augmentation': False,
                'mixture_augmentation': False,
                'with_exploration': False,
                'n_classes':len(LABEL_VALUES),
                'n_bands':img.shape[-1],
                'ignored_labels':IGNORED_LABELS,
                'device': torch.device("cpu"if dispositivo<0 else "cuda:"+str(dispositivo))}
    #redefinir las etiquetas entre 0 y num_clases puesto que se ignorará la etiqueta 0
    if 0 in hyperparams["ignored_labels"]:
      gt=gt-1
      #se actualiza el valor de las etiquetas en la variable hyperparams
      etiquetas_actualizadas=[]
      for etiqueta_anterior in hyperparams["ignored_labels"]:
        etiquetas_actualizadas.append(etiqueta_anterior-1)
      hyperparams["ignored_labels"]=etiquetas_actualizadas
    model, optimizer, loss, hyperparams = get_model(modelo,hyperparams["device"], **hyperparams)
    hyperparams["dataset"]="IndianPines"
    train_gt, test_gt = sample_gt(gt,
                                  hyperparams["training_sample"],
                                  mode=hyperparams["sampling_mode"])
    train_gt, val_gt = sample_gt(train_gt, 0.8, mode="random")
    
    # Generate the dataset

    train_dataset = HyperX(img, train_gt, **hyperparams)

    val_dataset = HyperX(img, val_gt, **hyperparams)
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=False
    )
    test_dataset=HyperX(img,test_gt,**hyperparams)
    test_loader=DataLoader(
        test_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=False
    )
    canales=img.shape[-1]
    img=torch.tensor(img,device="cpu")
    mean=torch.mean(img)
    std=torch.std(img)
    num_clases=len(torch.unique(torch.tensor(train_dataset.labels)))
    return (
        canales,
        num_clases,
        train_dataset,
        test_dataset,
        val_dataset,
        test_loader,
        val_loader,
        model,
        optimizer,
        loss,
        hyperparams)
def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('DC error: loop hyper-parameters are not defined for %d ipc'%ipc)
    return outer_loop, inner_loop
def get_images(c, n,indices_class,images_all): # get random n images from class c
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return images_all[idx_shuffle]
def epoch(mode, dataloader, net, optimizer, criterion, device):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(device)
    criterion = criterion.to(device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(device)
        lab = datum[1].long().to(device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg
def evaluate_synset(net, images_train, labels_train, testloader, learningrate, batchsize_train, device, Epoch = 600):
    net = net.to(device)
    images_train = images_train.to(device)
    labels_train = labels_train.to(device)
    lr = float(learningrate)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=batchsize_train, shuffle=False, num_workers=0)

    for ep in range(Epoch+1):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, device)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, device)
    print('Evaluate: epoch = %04d, train loss = %.6f, train acc = %.4f, test acc = %.4f' % (Epoch, loss_train, acc_train, acc_test))

    return net, acc_train, acc_test
def graficar(firmas,etiquetas,num_classes=None,figsize=(35,15),filas=None,columnas=None):
    if num_classes==None:
        num_classes=len(torch.unique(etiquetas))
    if filas==None or columnas==None:
        #determinar cuantas filas y columnas tendrán los subplots por fuerza bruta
        filas=int(math.ceil(num_classes**0.5))#inicializar la cantidad de filas
        columnas=num_classes/filas#filas*columnas=num_classes
        while columnas!=int(columnas):#mientras columnas no sea un número entero
            filas=filas-1
            columnas=num_classes/filas
        columnas=int(columnas)
    fig,axs=plt.subplots(filas,columnas,figsize=figsize)
    if filas!=1:
        etiquetas_unicas=torch.unique(etiquetas)
        #matriz bidimensional que asigna a cada etiqueta una posicion i,j que corresponde a la posicion i,j del subplot en el que se graficaran las firmas de su clase, se inicializa como un vector con etiquetas inexistentes
        ind=torch.ones(filas*columnas)+int(torch.max(etiquetas_unicas))
        #se guarda la etiqueta que tendran los subplots vacios
        etiqueta_vacio=int(ind[0])

        ind[:num_classes]=etiquetas_unicas
        ind=torch.reshape(ind,(filas,columnas))
        for ind_firma in range(len(etiquetas)):
            #se busca la posicion i,j del subplot en el que debe ir la firma en el indice ind_firma
            i,j=(ind==etiquetas[ind_firma]).nonzero(as_tuple=True)
            i=int(i)
            j=int(j)
            axs[i,j].plot(firmas[ind_firma])
        #poner a cada subplot la etiqueta a la que corresponde como subtitulo
        for i in range(filas):
            for j in range(columnas):
                if int(ind[i,j])!=etiqueta_vacio:
                    axs[i,j].set_title(int(ind[i,j]))
    else:
        #vector que indica el orden en que se ubicarán los subplots de las etiquetas
        ind=torch.unique(etiquetas)
        for ind_firma in range(len(etiquetas)):
            axs[int((ind==etiquetas[ind_firma]).nonzero())].plot(firmas[ind_firma])
        #poner a cada subplot su respectivo subtitulo
        for i in range(columnas):
            axs[i].set_title(int(ind[i]))
    plt.show()
"""
def aumento(img_orig,tecnica,fact_aum=None):
    #img_orig: tensor del lote de imagenes a aumetar
    #fact_aum: factor de aumento (ver ayuda del archivo ejecutarDestilado.py)
    #tecnica: nombre de la tecnica usar
    #dispositivo: dispositivo de pytorch en el que se alojarán los tensores
    #inicializacion de tensores aumentados
    #tamaño de lote aumentado
    if fact_aum>0:
        if tecnica=="escalamiento":
            maxi=torch.max(img_orig)
            if maxi!=1:
                fact_aum=1
            else: 
                fact_aum=0
                warnings.warn("Los datos reales ya están entre 0 y 1, no hay nada que escalar")
        dim_img_aum=list(img_orig.shape)
        dim_img_aum[0]=int(dim_img_aum[0]*(1+fact_aum))
        #inicializar tensor de imagenes aumentadas
        dispositivo=torch.device(f"cuda:{img_orig.get_device()}"if img_orig.get_device()>=0 else "cpu")
        img_aum=torch.empty(dim_img_aum,device=dispositivo)
        #recorrer tensores aumentados para llenarlos
        ind_aum=0#indice actual de los tensores aumentados
        ind_orig=0#indice actual en el tensor de datos no aumentados
        while ind_aum<dim_img_aum[0]:
            #se asigna la pareja de datos original
            img_aum[ind_aum]=img_orig[ind_orig]
            #se asignan los datos nuevos
            ind_aum+=1
            for i in range(fact_aum):
                if tecnica=="ruido":
                    img_aum[ind_aum]=torch.clip(img_orig[ind_orig]+torch.rand(img_orig[ind_orig].shape,device=dispositivo)/10-0.05,0,1)
                elif tecnica=="escalamiento":
                    img_aum[ind_aum]=img_orig[ind_orig]/maxi
                else:
                    exit(f"Tecnica de aumento {tecnica} no conocida")
                ind_aum+=1
            #ind_orig=ind_orig+1
        img_aum.requires_grad_(img_orig.requires_grad)
    else:
        img_aum=img_orig
    return img_aum
"""
#hacer aumento sumando ruido
def adicion(imgs,fact_aum,etq=None):
    copia=imgs
    copia_etq=etq
    for _ in range(fact_aum):
        #sumar un ruido en el intervalo [-0.5,0.5]
        imgs=torch.cat((imgs,copia+torch.rand(copia.shape)-0.5))
        if etq!=None:
            etq=torch.cat((etq,copia_etq))
    if etq==None:
        return torch.clip(imgs,0,1)
    else:
        return torch.clip(imgs,0,1),etq
#multiplicar tensor por un escalar
def multiplicacion(escalar,tensor):
    return escalar*tensor
#hacer aumento multiplicando o elevando a numeros aleatorios
def noAdicion(imgs,parametros,metodo,etq=None):
    copia=imgs
    if metodo=="potencia":
        nuevosDatos=torch.pow
    elif metodo=="escalamiento":
        nuevosDatos=multiplicacion
    #operacion matematica que aplica el aumento
    for parametro in parametros:
        imgs=torch.cat((imgs,nuevosDatos(copia,parametro)))
    if etq==None:
        return imgs
    else:
        return imgs,torch.repeat_interleave(etq,len(parametros),0)
def aumento(tecnica,replicas,img,etq=None):
    #tecnica: método de aumento
    #replicas: cantidad de replicas por cada dato
    #img: imagenes sin aumentar
    #etq: etiquetas sin aumentar
    #funciones que generarán cada nueva muestra según la técnica
    funcion={
        "ruido":lambda tensor:tensor+torch.rand(tensor.shape)-0.5,
        "escalamiento":lambda tensor:tensor*(torch.rand(1).item()/4+0.85),
        "potencia":lambda tensor:torch.pow(tensor,torch.ran(1).item()/4+0.85)
    }
    for i in range(len(img)):
        for _ in range(replicas):
            img=torch.cat((img,torch.unsqueeze(funcion[tecnica](img[i]),0)))
            if etq!=None:
                etq=torch.cat((etq,torch.unsqueeze(etq[i],0)))
    return img if etq==None else (img,etq)