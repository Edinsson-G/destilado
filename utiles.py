import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import math
import os
import spectral
from scipy import io, misc
import sklearn.model_selection
class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]
def get_images(c, n,indices_class,images_all): # get random n images from class c
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return images_all[idx_shuffle]

def graficar(firmas,etiquetas,num_classes=None,figsize=(35,15),filas=None,columnas=None,ruta=None,titulo=None):
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
    
    _,axs=plt.subplots(filas,columnas,figsize=figsize)
    if titulo!=None:
        plt.suptitle(titulo, figure=plt.gcf(),fontsize=50)
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
    if ruta!=None:
        plt.savefig(ruta)
        plt.close()
    else:
        plt.show()
def aumento(funcion,replicas,img,etq=None):
    #funcion: función de python que reciba cómo argumento unicamente una muestra y retorne una versión modificada de esa muestra para el aumento de datos
    #replicas: cantidad de replicas por cada dato
    #img: imagenes sin aumentar
    #etq: etiquetas sin aumentar
    #funciones que generarán cada nueva muestra según la técnica
    for i in range(len(img)):
        for _ in range(replicas):
            img=torch.cat((img,torch.unsqueeze(funcion(img[i]),0)))
            if etq!=None:
                etq=torch.cat((etq,torch.unsqueeze(etq[i],0)))
    return img if etq==None else (img,etq)
def embebido(red,entrada):
    #entrada=torch.unsqueeze(entrada,1)
    for capa in list(red.children())[:-2]:
        entrada=capa(entrada)
    return entrada.view(entrada.size(0),-1)
def sample_gt(gt, train_size, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)

    if mode == 'random':
       train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y)
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[train_indices] = gt[train_indices]
       test_gt[test_indices] = gt[test_indices]
    elif mode == 'fixed':
       print("Sampling {} with train size = {}".format(mode, train_size))
       train_indices, test_indices = [], []
       for c in np.unique(gt):
           if c == 0:
              continue
           indices = np.nonzero(gt == c)
           X = list(zip(*indices)) # x,y features

           train, test = sklearn.model_selection.train_test_split(X, train_size=train_size)
           train_indices += train
           test_indices += test
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[train_indices] = gt[train_indices]
       test_gt[test_indices] = gt[test_indices]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / (first_half_count + second_half_count)
                    if ratio > 0.9 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt
def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return misc.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))