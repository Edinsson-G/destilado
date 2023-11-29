import torch
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import argparse
from datos import graficar
parser=argparse.ArgumentParser(description="Graficar función de perdida")
parser.add_argument("--subcarpeta",type=str,required=True)
ruta="resultados/"+parser.parse_args().subcarpeta+'/'
perdida=torch.load(ruta+"histPerdida.pt")
plt.plot(perdida)
plt.title(parser.parse_args().subcarpeta)
plt.xlabel("iteración")
plt.ylabel("pérdida")
plt.show()
#graficar las firmas de menor perdida
graficar(torch.load(ruta+"imgs.pt")[perdida.index(min(perdida))].detach().to("cpu"),
         torch.load(ruta+"label_syn.pt"))
#graficar firmas reales
graficar(torch.load(ruta+"images_all.pt"),torch.load(ruta+"labels_all.pt"))