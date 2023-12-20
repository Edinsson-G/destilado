import os
import matplotlib.pyplot as plt
import torch
import copy
from datos import graficar
os.chdir("resultados")
#recolectar los accuracies de testeo
carpetas=os.listdir()
acc=torch.empty(len(carpetas),dtype=torch.float16)
lr=copy.deepcopy(acc)
perdidas=[]
minPerd=copy.deepcopy(acc)#minimo valor de perdida
for ind,carpeta in enumerate(carpetas):
    lr[ind]=float(carpeta.split(' ')[9])
    acc[ind]=torch.load(carpeta+"/accTestDatosdestilados.pt")
    perdidas.append(torch.load(carpeta+"/histPerdida.pt"))
    minPerd[ind]=perdidas[-1][perdidas[-1].index(min(perdidas[-1]))]
plt.bar(lr,acc,label="accuracy test")
plt.bar(minPerd,label="minima perdida")
plt.xlabel("lr")
plt.ylabe("accuracy")
plt.title("Accuracies de testeo")
plt.show()
#graficar funcion de perdida en cada caso
graficar(perdidas,lr)