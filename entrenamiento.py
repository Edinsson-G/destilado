from ignite.metrics import Accuracy
import torch
from perdida import *
from datos import *
from tqdm import tqdm
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
def train(
    net,
    optimizer,
    criterion,
    data_loader,
    epoch,
    test_loader=None,
    val_loader=None,
    device=torch.device("cpu")
):
    net.to(device)
    net.train()
    acc_ent_hist=[]
    acc_val_hist=[]
    perdida=[]
    for e in tqdm(range(1, epoch + 1)):
        # Set the network to training mode
        net.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        test_loss = AverageMeter()
        test_acc = AverageMeter()
        accuracy=Accuracy()
        
        data_loop_train = tqdm(enumerate(data_loader), total=len(data_loader), colour='red')
        acc_ent_iter=[]
        loss_iter=[]
        for batch_idx, (data, target) in data_loop_train:
            # print(target)

            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target.type(torch.long).to(device))
            loss.backward()
            optimizer.step()
            accuracy.update((output,target))
            correct = accuracy.compute()
            #entr_acc=float(correct)
            acc_ent_iter.append(float(correct))
            train_loss.update(loss.item(), data.size(0))

            train_acc.update(correct, data.size(0))
            
            dict_metrics = dict(loss=train_loss.avg,acc=train_acc.avg)


            data_loop_train.set_description(f'Network Training [{e} / {epoch}]')
            data_loop_train.set_postfix(**dict_metrics)
            loss_iter.append(train_loss.avg)
            accuracy.reset()
        acc_ent_hist.append(torch.mean(torch.tensor(acc_ent_iter)))
        perdida.append(torch.mean(torch.tensor(loss_iter)))
        if val_loader!=None:
            data_loop_test = tqdm(enumerate(val_loader), total=len(val_loader), colour='green')
            #data_loop_test = enumerate(test_loader)

            net.eval()
            # Run the testing loop for one epoch
            for batch_idx, (data, target) in data_loop_test:

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
    if test_loader!=None:
        #calcular accuracy de testeo
        for img,etq in test_loader:
            img=img.to(device)
            etq=etq.to(device)
            accuracy.update((net(img),etq))
            acc_test=float(accuracy.compute())
            accuracy.reset()
        print("accuracy de testeo:",acc_test)
    if test_loader!=None and val_loader!=None:
        return net,perdida,acc_ent_hist,acc_val_hist,acc_test
    else:
        return net
"""
def retroporopagacion(data,target,optimizer,net,device,criterion):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = net(data)
    loss = criterion(output, target.type(torch.long).to(device))
    loss.backward()
    optimizer.step()
def train(net,optimizer,criterion,data_loader,device,epoch=100,test_loader=None,val_loader=None):
    net.to(device)
    ciclo=tqdm(range(1+epoch+1))
    if test_loader==None:
        net.train()
        for _ in ciclo:
            train_loss = AverageMeter()
            for data, target in tqdm(data_loader, total=len(data_loader), colour='red'):
                retroporopagacion(data,target,optimizer,net,device,criterion)
"""