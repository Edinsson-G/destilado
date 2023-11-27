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
    test_loader,
    dst_val=None,
    #scheduler=None,
    device=torch.device("cpu"),
    mostrar_avances=False
):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)
    net.train()

    # losses = np.zeros(1000000)
    # mean_losses = np.zeros(100000000)

    iter_ = 1
    loss_win = None
    #si no se suministran datos de validación no se calculará ningún accuracy (ni si quiera para los datos de entrenamiento)
    if dst_val is not None:
      val_accuracy=Accuracy()
      accuracy=Accuracy()
    #for e in tqdm(range(1, epoch + 1), desc="Training the network"):
    print('Start training ...')
    for e in tqdm(range(1, epoch + 1)):
        # Set the network to training mode
        net.train()
        avg_loss = 0.0
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        test_loss = AverageMeter()
        test_acc = AverageMeter()
        accuracy=Accuracy()

        data_loop_train = tqdm(enumerate(data_loader), total=len(data_loader), colour='red')
        #data_loop_train =enumerate(data_loader)

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
            entr_acc=float(correct)
            train_loss.update(loss.item(), data.size(0))

            train_acc.update(correct, data.size(0))

            dict_metrics = dict(loss=train_loss.avg,acc=train_acc.avg)


            data_loop_train.set_description(f'Network Training [{e} / {epoch}]')
            data_loop_train.set_postfix(**dict_metrics)
            accuracy.reset()



        data_loop_test = tqdm(enumerate(test_loader), total=len(test_loader), colour='green')
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
            test_acc1=float(correct)
            dict_metrics = dict(loss_test=test_loss.avg,acc_test=test_acc.avg)

            data_loop_test.set_description(f'Network Testing [{e} / {epoch}]')
            data_loop_test.set_postfix(**dict_metrics)

            accuracy.reset()

    return entr_acc,test_acc1