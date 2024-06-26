# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
from tqdm import tqdm
from models import get_model
from utiles import sample_gt
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data import TensorDataset
import jax.random
from coreax import ArrayData, KernelHerding, RandomSample,SquaredExponentialKernel
from coreax.kernel import median_heuristic
from coreax.reduction import MapReduce,SizeReduce
import math
try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

from utiles import open_file

DATASETS_CONFIG = {
    "PaviaC": {
        "urls": [
            "http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat",
            "http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat",
        ],
        "img": "Pavia.mat",
        "gt": "Pavia_gt.mat",
    },
    "Salinas": {
        "urls": [
            "http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
            "http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
        ],
        "img": "Salinas_corrected.mat",
        "gt": "Salinas_gt.mat",
    },
    "PaviaU": {
        "urls": [
            "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
            "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
        ],
        "img": "PaviaU.mat",
        "gt": "PaviaU_gt.mat",
    },
    "KSC": {
        "urls": [
            "http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat",
            "http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat",
        ],
        "img": "KSC.mat",
        "gt": "KSC_gt.mat",
    },
    "IndianPines": {
        "urls": [
            "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
            "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
        ],
        "img": "Indian_pines_corrected.mat",
        "gt": "Indian_pines_gt.mat",
    },
    "Botswana": {
        "urls": [
            "http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat",
            "http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat",
        ],
        "img": "Botswana.mat",
        "gt": "Botswana_gt.mat",
    },
}

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG

    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    """Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None

    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    folder = target_folder + datasets[dataset_name].get("folder", dataset_name + "/")
    if dataset.get("download", True):
        # Download the dataset if is not present
        if not os.path.isdir(folder):
            os.makedirs(folder)
        for url in datasets[dataset_name]["urls"]:
            # download the files
            filename = url.split("/")[-1]
            if not os.path.exists(folder + filename):
                with TqdmUpTo(
                    unit="B",
                    unit_scale=True,
                    miniters=1,
                    desc="Downloading {}".format(filename),
                ) as t:
                    urlretrieve(url, filename=folder + filename, reporthook=t.update_to)
    elif not os.path.isdir(folder):
        print("WARNING: {} is not downloadable.".format(dataset_name))

    if dataset_name == "PaviaC":
        # Load the image
        img = open_file(folder + "Pavia.mat")["pavia"]

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + "Pavia_gt.mat")["pavia_gt"]

        label_values = [
            "Undefined",
            "Water",
            "Trees",
            "Asphalt",
            "Self-Blocking Bricks",
            "Bitumen",
            "Tiles",
            "Shadows",
            "Meadows",
            "Bare Soil",
        ]

        ignored_labels = [0]

    elif dataset_name == "PaviaU":
        # Load the image
        img = open_file(folder + "PaviaU.mat")["paviaU"]

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + "PaviaU_gt.mat")["paviaU_gt"]

        label_values = [
            "Undefined",
            "Asphalt",
            "Meadows",
            "Gravel",
            "Trees",
            "Painted metal sheets",
            "Bare Soil",
            "Bitumen",
            "Self-Blocking Bricks",
            "Shadows",
        ]

        ignored_labels = [0]

    elif dataset_name == "Salinas":
        img = open_file(folder + "Salinas_corrected.mat")["salinas_corrected"]

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + "Salinas_gt.mat")["salinas_gt"]

        label_values = [
            "Undefined",
            "Brocoli_green_weeds_1",
            "Brocoli_green_weeds_2",
            "Fallow",
            "Fallow_rough_plow",
            "Fallow_smooth",
            "Stubble",
            "Celery",
            "Grapes_untrained",
            "Soil_vinyard_develop",
            "Corn_senesced_green_weeds",
            "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk",
            "Lettuce_romaine_6wk",
            "Lettuce_romaine_7wk",
            "Vinyard_untrained",
            "Vinyard_vertical_trellis",
        ]

        ignored_labels = [0]

    elif dataset_name == "IndianPines":
        # Load the image
        img = open_file(folder + "Indian_pines_corrected.mat")
        img = img["indian_pines_corrected"]

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + "Indian_pines_gt.mat")["indian_pines_gt"]
        label_values = [
            "Undefined",
            "Alfalfa",
            "Corn-notill",
            "Corn-mintill",
            "Corn",
            "Grass-pasture",
            "Grass-trees",
            "Grass-pasture-mowed",
            "Hay-windrowed",
            "Oats",
            "Soybean-notill",
            "Soybean-mintill",
            "Soybean-clean",
            "Wheat",
            "Woods",
            "Buildings-Grass-Trees-Drives",
            "Stone-Steel-Towers",
        ]

        ignored_labels = [0]

    elif dataset_name == "Botswana":
        # Load the image
        img = open_file(folder + "Botswana.mat")["Botswana"]

        rgb_bands = (75, 33, 15)

        gt = open_file(folder + "Botswana_gt.mat")["Botswana_gt"]
        label_values = [
            "Undefined",
            "Water",
            "Hippo grass",
            "Floodplain grasses 1",
            "Floodplain grasses 2",
            "Reeds",
            "Riparian",
            "Firescar",
            "Island interior",
            "Acacia woodlands",
            "Acacia shrublands",
            "Acacia grasslands",
            "Short mopane",
            "Mixed mopane",
            "Exposed soils",
        ]

        ignored_labels = [0]

    elif dataset_name == "KSC":
        # Load the image
        img = open_file(folder + "KSC.mat")["KSC"]

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + "KSC_gt.mat")["KSC_gt"]
        label_values = [
            "Undefined",
            "Scrub",
            "Willow swamp",
            "Cabbage palm hammock",
            "Cabbage palm/oak hammock",
            "Slash pine",
            "Oak/broadleaf hammock",
            "Hardwood swamp",
            "Graminoid marsh",
            "Spartina marsh",
            "Cattail marsh",
            "Salt marsh",
            "Mud flats",
            "Wate",
        ]

        ignored_labels = [0]
    else:
        # Custom dataset
        (
            img,
            gt,
            rgb_bands,
            ignored_labels,
            label_values,
            palette,
        ) = CUSTOM_DATASETS_CONFIG[dataset_name]["loader"](folder)

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print(
            "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled."
        )
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype="float32")
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img, gt, label_values, ignored_labels, rgb_bands, palette


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.name = hyperparams["dataset"]
        self.patch_size = hyperparams["patch_size"]
        self.ignored_labels = set(hyperparams["ignored_labels"])
        self.flip_augmentation = hyperparams["flip_augmentation"]
        self.radiation_augmentation = hyperparams["radiation_augmentation"]
        self.mixture_augmentation = hyperparams["mixture_augmentation"]
        self.center_pixel = hyperparams["center_pixel"]
        supervision = hyperparams["supervision"]
        # Fully supervised : use all pixels with label not ignored
        if supervision == "full":
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == "semi":
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array(
            [
                (x, y)
                for x, y in zip(x_pos, y_pos)
                if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p
            ]
        )
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)
    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1.0, size=2)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert self.labels[l_indice] == value
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.1:
            data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.2:
            data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
        label = np.asarray(np.copy(label), dtype="int64")

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)
        return data, label
#funcion para extraer un conjunto coreset
def coreset(images_all,indices_class,ipc,muestreo,escalable,semilla):
    tam=list(images_all.shape)
    tam[0]=len(indices_class)*ipc
    dispositivo=images_all.device
    img_cor=np.empty(tam)
    images_all=images_all.cpu().numpy()
    if muestreo=="herding":
        for clase in range(len(indices_class)):
            #filtrar imagenes de esta clase
            img_clase=images_all[indices_class[clase]]
            #vectorizar la imagen si no está vectorizada
            dim=list(img_clase.shape)
            if len(dim)>2:
                img_clase=np.reshape(
                    img_clase,
                    (dim[0],math.prod(dim[1:]))
                )
            #entrenar el método de herding
            length_scale=median_heuristic(
                img_clase[np.random.default_rng(semilla).choice(
                    len(indices_class[clase]),
                    min(len(indices_class[clase]), 1_000),
                    replace=len(indices_class[clase])<ipc
                )]
            )
            obj_coreset=KernelHerding(
                jax.random.key(semilla),
                kernel=SquaredExponentialKernel(length_scale=length_scale if length_scale>0 else 0.01)
            )
            obj_coreset.fit(
                original_data=ArrayData.load(img_clase),
                strategy=MapReduce(coreset_size=ipc, leaf_size=200)if escalable else SizeReduce(coreset_size=ipc)
            )
            i=clase*ipc
            img_cor[i:i+ipc]=images_all[indices_class[clase]][obj_coreset.coreset_indices]
    else:
        #muestreo uniforme
        for clase in range(len(indices_class)):
            img_clase=images_all[indices_class[clase]]
            obj_coreset=RandomSample(jax.random.key(semilla),unique=False)
            obj_coreset.fit(original_data=ArrayData.load(img_clase),strategy=MapReduce(coreset_size=ipc, leaf_size=200)if escalable else SizeReduce(coreset_size=ipc))
            i=clase*ipc
            img_cor[i:i+ipc]=obj_coreset.coreset
    return torch.tensor(img_cor,dtype=torch.float32,device=dispositivo)
def datosYred(modelo,conjunto,dispositivo):
    img,gt,_,IGNORED_LABELS,_,_= get_dataset(conjunto,"Datasets/")
    gt=np.array(gt,dtype=np.int32)
    hiperparametros={
        'dataset':conjunto,
        'model':modelo,
        'folder':'./Datasets/',
        'cuda':"cpu"if dispositivo<0 else f"cuda:{dispositivo}",
        'runs': 1,
        'training_sample': 0.99,
        'sampling_mode': 'fixed',
        'class_balancing': True,
        'test_stride': 1,
        'flip_augmentation': False,
        'radiation_augmentation': False,
        'mixture_augmentation': False,
        'with_exploration': False,
        'n_classes':np.unique(gt).size,
        'n_bands':img.shape[-1],
        'ignored_labels':IGNORED_LABELS,
        'device': dispositivo
    }
    clases=np.unique(gt)
    num_classes=clases.size
    for etiqueta_ingnorada in hiperparametros["ignored_labels"]:
        if etiqueta_ingnorada in clases:
            num_classes=num_classes-1
    hiperparametros["n_classes"]=num_classes
    red,optimizador_red,criterion,hiperparametros= get_model(hiperparametros["model"],hiperparametros["cuda"],**hiperparametros)
    train_gt,test_gt=sample_gt(gt,
                               hiperparametros["training_sample"],
                               mode=hiperparametros["sampling_mode"])
    #train_gt, val_gt = sample_gt(train_gt, 0.8, mode="random")
    #redefinir las etiquetas entre 0 y num_clases puesto que se ignorará la etiqueta 0
    dst=HyperX(img, train_gt, **hiperparametros)
    imagenes =  torch.cat([torch.unsqueeze(dst[i][0], dim=0) for i in range(len(dst))],dim=0) # Save the images (1,1,28,28)
    etiquetas = torch.tensor([int(dst[i][1]) for i in range(len(dst))],device=hiperparametros["cuda"]) # Save the labels
    #revolver
    n=len(etiquetas)
    for a in range(n):
        b=torch.randint(0,n,(1,)).item()
        #intercambiar imagen
        temp=imagenes[a]
        imagenes[a]=imagenes[b]
        imagenes[b]=temp
        #intercambiar etiqueta
        temp=etiquetas[a]
        etiquetas[a]=etiquetas[b]
        etiquetas[b]=temp
    if 0 in hiperparametros["ignored_labels"]:
      etiquetas=etiquetas-1
      hiperparametros["ignored_labels"]=(
          torch.tensor(hiperparametros["ignored_labels"])-1
          ).tolist()
    #dst_test=HyperX(img,test_gt,**hiperparametros)
    n_test=int(n*0.2)
    n_train=int((n-n_test)*0.8)
    dst_train=TensorDataset(imagenes[:n_train],etiquetas[:n_train])
    dst_test=TensorDataset(imagenes[n_train:n_train+n_test],etiquetas[n_train:n_train+n_test])
    dst_val=TensorDataset(imagenes[n_train+n_test:],etiquetas[n_train+n_test:])
    test_loader=DataLoader(dst_test,batch_size=len(dst_test),shuffle=True)
    #dst_val=HyperX(img, val_gt, **hiperparametros)
    val_loader= DataLoader(dst_val,batch_size=len(dst_val),shuffle=True)
    return dst_train,test_loader,val_loader,red,optimizador_red,criterion,hiperparametros
def vars_all(dst_train,n_clases):
    #preprocesar datos reales
    images_all = []
    labels_all = []
    indices_class = [[] for _ in range(n_clases)]
    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))] # Save the images (1,1,28,28)
    labels_all = [int(dst_train[i][1]) for i in range(len(dst_train))] # Save the labels
    for i, lab in enumerate(labels_all): # Save the index of each class labels
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0) # Cat images along the batch dimension
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=images_all.device) # Make the labels a tensor
    return images_all,labels_all,indices_class