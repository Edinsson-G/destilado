import torch
ruta="resultados/Modelo nn conjunto IndianPines ipc 10 ritmo de aprendizaje 1.0 aumento ruido/"
img=torch.load(ruta+"images_all.pt")
etq=torch.load(ruta+"labels_all.pt")
#imagenes por clase
img_pc=[]
for etq_i in torch.unique(etq):
    for i in range(len(etq)):
        if etq[i]==etq_i:
            img_pc.append(list(img[i]))
    print("clase:",etq_i.item())
    print("media:",torch.mean(torch.tensor(img_pc)).item())
    print("desviacion:",torch.std(torch.tensor(img_pc)).item(),"\n")
    img_pc=[]