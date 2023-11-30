import argparse
import torch
import os
from datos import obtener_datos
from entrenamiento import train
from torch.utils.data import TensorDataset,DataLoader
import secrets
parser=argparse.ArgumentParser(description="realizar entrenamientos para comprobar el funcionamiento del destilado.")
parser.add_argument("--carpetaAnterior",
                    type=str,
                    required=True,
                    help="Nombre de la carpeta en la que se almacenan los datos que se utilizarán para el entrenamiento, debe estar dentro de la carpeta resultados.")
parser.add_argument("--carpetaDestino",
                    type=str,
                    default=parser.parse_args().carpetaAnterior,
                    help="Carpeta en la que se guardarán los registros de este entrenamiento")
parser.add_argument("--dispositivo",
                    type=int,
                    default=-1,
                    help="Indice del dispositivo en el que se realizará el destilado, si es negativo se usará la CPU.")
parser.add_argument("--semilla",
                    type=int,
                    default=18,
                    help="semilla pseudoaleatoria para inicializar los pesos.")
parser.add_argument("--tipoConjunto",
                    type=str,
                    choices=["real","sintetico"],
                    default="sintetico",
                    help="Especificar si se realizará el entrenamiento con los datos reales (archivo images_all.pt y labels_all.pt) o sintéticos (imgs.pt y label_syn.pt)")
torch.manual_seed(parser.parse_args().semilla)
dispositivo=torch.device("cpu" if parser.parse_args().dispositivo<0 else "cuda:"+str(parser.parse_args().dispositivo))
print("Algoritmo ejecutándose en",dispositivo,"\n\n")
carpeta="resultados/"
ruta_anterior=carpeta+parser.parse_args().carpetaAnterior+'/'
#carguar datos
imgs=torch.load(ruta_anterior+"imgs",map_location=dispositivo)
if type(imgs)==list:
    imgs=imgs[-1]
imgs=imgs.detach()
#Carguar modelo
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
hiperparametros)=obtener_datos(parser.parse_args().conjunto,
                                parser.parse_args().dispositivo,
                                parser.parse_args().modelo)
del dst_train,dst_test,dst_val,train_loader,test_loader,val_loader
#media de los datos destilados
med_dest=torch.mean(imgs)
#desviación estandar de los datos destilados
std_dest=torch.std(imgs)
for leyenda,valor in zip(["media de los datos reales:",
                          "media de los datos sintéticos:",
                          "diferencia entre medias:",
                          "desviación estandar de las imagenes reales:",
                          "desviación estandar de las imagenes sintéticas:",
                          "diferencia entre desviaciones:"],
                          [mean,med_dest,abs(med_dest-mean),std,std_dest,abs(std-std_dest)]):
    print(leyenda,valor.item())
print("\n\n")
del mean,std,med_dest,std_dest
 #crear carpeta destino si no existe
ruta_destino=carpeta+parser.parse_args().carpetaDestino+'/'
if parser.parse_args().carpetaDestino not in os.listdir(carpeta):
    os.mkdir(ruta_destino)
if "acc_test"not in os.listdir(ruta_destino):
    print("Iniciando entrenamiento con los datos destilados")
    #se guarda el modelo recien inicializado para el entrenamiento con las firmas muestreadas
    torch.save(net,ruta_destino+"red_inicializada")
    torch.save(optimizer_net,ruta_destino+"optimizador_inicializado")
    acc_train,acc_test=train(net,
                            optimizer_net,
                            criterion,
                            DataLoader(TensorDataset(imgs,
                                                    torch.load(ruta_anterior+"label_syn",
                                                                map_location=dispositivo)),
                                        batch_size=hiperparametros["batch_size"],shuffle=False,num_workers=0),
                            1000,
                            test_loader=torch.load(ruta_anterior+"test_loader",
                                                    map_location=dispositivo),
                            device=dispositivo)
    if parser.parse_args().carpetaDestino!=None:
        #guardar registros
        for variable,archivo in zip((net,optimizer_net,acc_train,acc_test),
                                    ["net","optimizer_net","acc_train","acc_test"]):
            torch.save(variable,ruta_destino+archivo)
        print("Registros guardados en",ruta_destino)
else:
    print("Entrenmiento con datos destilados realizado con anterioridad.")
if ("acc_test_primer"not in os.listdir(ruta_destino)) or ("red_inicializada"not in os.listdir(ruta_destino)):
    print("\n\nIniciando entrenamiento con datos recién muestreados.")
    images_all=torch.load(ruta_anterior+"images_all")
    indices_class=torch.load(ruta_anterior+"indices_class")
    labels_all=torch.load(ruta_anterior+"labels_all")
    ipc=int(len(imgs)/num_classes)
    label_syn=torch.repeat_interleave(torch.arange(num_classes,requires_grad=False),int(ipc))
    #imagenes sintéticas
    tam=(num_classes*ipc,channel)
    if hiperparametros["patch_size"]>1:
        tam=tam+(hiperparametros["patch_size"],hiperparametros["patch_size"])
    image_syn=torch.empty(tam,device=dispositivo)
    for i,clase in enumerate(label_syn):
        image_syn[i]=images_all[secrets.choice(indices_class[clase])]
    #reiniciar modelo y optimizador para repetir entrenamiento con datos recien muestreados
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
    hiperparametros)=obtener_datos(parser.parse_args().conjunto,
                                    parser.parse_args().dispositivo,
                                    parser.parse_args().modelo)
    #se carga la misma inicializacion de pesos que se hizo para los datos destilados
    """
    if "red_inicializada"in os.listdir(ruta_destino):
        net=torch.load(ruta_destino+"red_inicializada")
        optimizer_net=torch.load(ruta_destino+"optimizador_inicializado")
    else:
        torch.save(net,ruta_destino+"red_inicializada")
        torch.save(optimizer_net,ruta_destino+"optimizador_inicializado")
    """
    del channel,dst_train,dst_test,dst_val,train_loader,test_loader,val_loader,num_classes
    acc_train,acc_test=train(net,
                            optimizer_net,
                            criterion,
                            DataLoader(TensorDataset(image_syn,label_syn),
                                        batch_size=hiperparametros["batch_size"],
                                        shuffle=False),
                            1000,
                            test_loader=torch.load(ruta_anterior+"test_loader",
                                                    map_location=dispositivo),
                            device=dispositivo)
    if parser.parse_args().carpetaDestino!=None:
        torch.save(acc_train,ruta_destino+"acc_ent_primer")
        torch.save(acc_test,ruta_destino+"acc_test_primer")
        torch.save(net,ruta_destino+"net_primer")
        torch.save(optimizer_net,ruta_destino+"optimizer_net")
        torch.save(image_syn,ruta_destino+"image_syn")
        torch.save(label_syn,ruta_destino+"label_syn")
    
else:
    print("Entrenamiento con datos recién muestreados realizado con anterioridad.")