from ignite.metrics import Accuracy
import torch
from tqdm import tqdm
import copy
import sys
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def retropropagacion(data,target,device,optimizer,net,criterion):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = net(data)
    loss = criterion(output, target.type(torch.long).to(device))
    loss.backward()
    optimizer.step()
    return optimizer,net,loss,criterion,output
def train(net,optimizer,criterion,data_loader,epoch=100,test_loader=None,val_loader=None,device=torch.device("cpu")):
    net.to(device)
    net.train()
    if val_loader==None:
        ciclo=tqdm(range(1,epoch+1),colour="red")
        #inicializar con el máximo valor flotante soportado por la máquina
        minPerd=sys.float_info.max#minima pérdida alzanzada hasta ahora
        net.train()
        for e in ciclo:
            perdida=AverageMeter()
            ciclo.set_description(f"Entrenamiento [{e}/{epoch}]")
            for data,target in data_loader:
                optimizer,net,loss,criterion,_=retropropagacion(data,target,device,optimizer,net,criterion)
                perdida.update(loss.item(),data.size(0))
                ciclo.set_postfix(**{"pérdida":perdida.avg})
            #guardar pesos con menor pérdida promedio
            if perdida.avg<minPerd:
                pesos=copy.deepcopy(net.state_dict())
                minPerd=perdida.avg
        net.eval()
        net.load_state_dict(pesos)
        retornar=(net,)
    else:
        acc_ent_hist=[]
        acc_val_hist=[]
        perdida=[]
        maxcc=-1
        #cantidad de epocas sin mejora
        estancamiento=0
        e=0
        while estancamiento<300 and e<epoch:
            net.train()
            train_loss = AverageMeter()
            train_acc = AverageMeter()
            test_loss = AverageMeter()
            test_acc = AverageMeter()
            accuracy=Accuracy()
            acc_ent_iter=[]
            loss_iter=[]
            for data,target in data_loader:
                optimizer,net,loss,criterion,output=retropropagacion(data,target,device,optimizer,net,criterion)
                accuracy.update((output.to(device),target.to(device)))
                correct = accuracy.compute()
                acc_ent_iter.append(float(correct))
                train_loss.update(loss.item(), data.size(0))

                train_acc.update(correct, data.size(0))
                
            dict_metrics = dict(loss=train_loss.avg)


            ciclo.set_description(f'Network Training [{e} / {epoch}]')
            ciclo.set_postfix(**dict_metrics)
            loss_iter.append(train_loss.avg)
            accuracy.reset()
            acc_ent_hist.append(torch.mean(torch.tensor(acc_ent_iter)))
            perdida.append(torch.mean(torch.tensor(loss_iter)))
            #iniciar con la validación
            data_loop_test = tqdm(val_loader, total=len(val_loader), colour='green')
            #data_loop_test = enumerate(test_loader)

            net.eval()
            # Run the testing loop for one epoch
            for data, target in data_loop_test:

                # Load the data into the GPU if required
                #data, target = data.to(device), target.to(device)-1
                data, target = data.to(device), target.to(device)
                output = net(data)
                loss = criterion(output, target.type(torch.long).to(device))
                test_loss.update(loss.item(), data.size(0))
                accuracy.update((output,target))
                correct = accuracy.compute()

                test_acc.update(correct, data.size(0))
                acc_val_hist.append(float(correct))
                #test_acc1=float(correct)
                dict_metrics = dict(loss_test=test_loss.avg,acc_test=test_acc.avg)

                data_loop_test.set_description(f'Network Testing [{e} / {epoch}]')
                data_loop_test.set_postfix(**dict_metrics)

                accuracy.reset()
            if correct>maxcc:
                #guardar los mejores pesos hasta el momento
                pesos=copy.deepcopy(net.state_dict())
                maxcc=correct
                estancamiento=0
            else:
                estancamiento+=1
            e=e+1
        if e<epoch:
            print("Interrupción temprana")
        #devolver los mejores pesos
        net.eval()
        net.load_state_dict(pesos)
        retornar=(net,perdida,acc_ent_hist,acc_val_hist)
    if test_loader!=None:
        #calcular accuracy de testeo
        net.eval()
        accuracy=Accuracy()
        for img,etq in test_loader:
            img=img.to(device)
            etq=etq.to(device)
            accuracy.update((net(img),etq))
            acc_test=float(accuracy.compute())
            accuracy.reset()
        retornar=retornar+(acc_test,)
    return retornar