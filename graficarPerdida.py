import torch
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import argparse
parser=argparse.ArgumentParser(description="Graficar función de perdida")
parser.add_argument("--subcarpeta",type=str,required=True)
plt.plot(torch.load("resultados/"+parser.parse_args().subcarpeta+"/histPerdida.pt"))
plt.title(parser.parse_args().subcarpeta)
plt.xlabel("iteración")
plt.ylabel("pérdida")
plt.show()
imgs=torch.load("resultados/"+parser.parse_args().subcarpeta+"/imgs.pt")
print(torch.unique(imgs[0]-imgs[1]))