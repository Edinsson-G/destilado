import torch
import secrets
ruta="resultados/Modelo nn conjunto IndianPines ipc 10/"
images_all = []
images_all = []
labels_all = []
indices_class = [[] for c in range(num_classes)]
images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))] # Save the images (1,1,28,28)
labels_all = [int(dst_train[i][1]) for i in range(len(dst_train))] # Save the labels
for i, lab in enumerate(labels_all): # Save the index of each class labels
indices_class[lab].append(i)
images_all = torch.cat(images_all, dim=0).to(device) # Cat images along the batch dimension
labels_all = torch.tensor(labels_all, dtype=torch.long, device=device) # Make the labels a tensor
#incializar con muestras aleatorias de cada clase 
#etiquetas sinteticas de la forma [0,0,1,1,1,...,num_classes,num_classes,num_classes] (la cantidad real de repeticiones las determina el ipc)
ipc=10
label_syn=torch.repeat_interleave(torch.arange(num_classes,requires_grad=False),ipc)
#imagenes sintéticas
tam=(num_classes*ipc,channel)
if hiperparametros["patch_size"]>1:
tam=tam+(hiperparametros["patch_size"],hiperparametros["patch_size"])
image_syn=torch.empty(tam,device=device)
del tam
for i,clase in enumerate(label_syn):
image_syn[i]=images_all[secrets.choice(indices_class[clase])]