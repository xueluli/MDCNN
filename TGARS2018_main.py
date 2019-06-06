import argparse, os, time
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import h5py, cv2
from dataset import DatasetFromHdf5
from MDCNN import *
import pdb

VIEW_PATH_LIST = ['view40-50','view80-100','view260-280','view310-320','view355-5']

parser = argparse.ArgumentParser(description="Pytorch TGARSS2018")
#parser.add_argument("--TrainData", default="/cvdata/NTIRE19/H5/18_VAL_512s64.h5", type=str, help="Training datapath")
parser.add_argument("--batchSize", type=int, default=32, help="Training batch size")
parser.add_argument("--ViewNumber", type=int, default=5, help="View number")
parser.add_argument("--TrainingSize", type=int, default=5, help="Training Size")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default=1")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate, Default=0.1")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default=1e-4")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train for")
parser.add_argument("--lr_decay_epoch", type=int, default=10, help="Number of epochs to train for")


def main():
    global opt, model
    opt = parser.parse_args()
    train_dataset_path = '/cvdata/xuelu/TGARS2019/H5/'+'view'+str(opt.ViewNumber)+'_'+'trainSize'+str(opt.TrainingSize)+'.h5'
    test_dataset_path = '/cvdata/xuelu/TGARS2019/H5/'+'test_'+'view'+str(opt.ViewNumber)+'.h5'
    print(train_dataset_path)
    print(test_dataset_path)
#    hf = h5py.File(dataset_path, 'r')
#    
#    input1 = hf.get(VIEW_PATH_LIST[0])
#    input2 = hf.get('Label')    
    
#    pdb.set_trace()
    cudnn.benchmark = True
    
    train_set = DatasetFromHdf5(train_dataset_path,opt.ViewNumber)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    test_set = DatasetFromHdf5(test_dataset_path,opt.ViewNumber)
    test_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)
    if opt.ViewNumber == 1:
       model = Network_view_1()
    
    if opt.ViewNumber == 2:
#       pdb.set_trace()
       model = Network_view_2()
    
    if opt.ViewNumber == 3:
       model = Network_view_3()
    
    if opt.ViewNumber == 4:
       model = Network_view_4()
    
    if opt.ViewNumber == 5:
       model = Network_view_5()
    
    criterion = nn.CrossEntropyLoss()
    
    cuda = opt.cuda
    if cuda  and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    if cuda:
       model = torch.nn.DataParallel(model).cuda()
       criterion = criterion.cuda()
    
    best_acc = 0.0
    for epoch in range(1, opt.num_epochs+1):
        lr = opt.lr * (0.60 ** (min(epoch-1, 200) // opt.lr_decay_epoch))
        optimizer = optim.Adam(model.parameters(), lr, weight_decay=opt.weight_decay)
        print('#################################################################')
        print('=> Training Epoch #%d, LR=%.10f' % (epoch, lr))
       
        epoch_loss = []
        num_correct = 0
        num_total = 0
        u = 0
        model.train()
        for batch_idx, (input, labels) in enumerate(training_data_loader):
            u = u + 1
#            labels = labels.flatten()
#            pdb.set_trace()
            if opt.ViewNumber == 1:
               input1 = Variable(input[0])
               if opt.cuda:
                  input1 = input1.cuda()
               output = model(input1)
            if opt.ViewNumber == 2:
               input1, input2 = Variable(input[0]), Variable(input[1])
               if opt.cuda:
                  input1 = input1.cuda()
                  input2 = input2.cuda()
               output = model(input1,input2)
#               pdb.set_trace()
            if opt.ViewNumber == 3:
               input1, input2, input3 = Variable(input[0]), Variable(input[1]), Variable(input[2])
               if opt.cuda:
                  input1 = input1.cuda()
                  input2 = input2.cuda()
                  input3 = input3.cuda()
               output = model(input1,input2,input3)
            if opt.ViewNumber == 4:
               input1, input2, input3, input4 = Variable(input[0]), Variable(input[1]), Variable(input[2]), Variable(input[3])
               if opt.cuda:
                  input1 = input1.cuda()
                  input2 = input2.cuda()
                  input3 = input3.cuda()
                  input4 = input4.cuda()             
               output = model(input1,input2,input3,input4)
            if opt.ViewNumber == 5:
               input1, input2, input3, input4, input5 = Variable(input[0]), Variable(input[1]), Variable(input[2]), Variable(input[3]), Variable(input[4])
               if opt.cuda:
                  input1 = input1.cuda()
                  input2 = input2.cuda()
                  input3 = input3.cuda()
                  input4 = input4.cuda()                  
                  input4 = input5.cuda()
               output = model(input1,input2,input3,input4,input5)
            if opt.cuda:
               labels = labels.cuda()
            
#            pdb.set_trace()
            loss = criterion(output, labels)
#            pdb.set_trace()
            
            print('| Epoch %2d Iter %3d\tBatch loss %.4f\t' % (epoch,u,loss))
            epoch_loss.append(loss.item())
            _, preds = torch.max(output.data, 1)
            num_total += labels.size(0)
            num_correct += torch.sum(preds == labels)            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_acc = 100 * num_correct / num_total
        test_acc, confusion = accuracy(model, test_data_loader, opt.ViewNumber)        
        
        if test_acc > best_acc:
           best_acc = test_acc
           best_confusion = confusion
           best_epoch = epoch
#           print('*', end='')
        print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%' %
                  (epoch, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))
    print('Best at epoch %d, test accuaray %f' % (best_epoch, best_acc))#        test_acc = self.
    print(best_confusion)
#        pdb.set_trace()
#            input, target = Variable(batch[0]), Variable(batch[1])
#            if opt.cuda:
#               input = input.cuda()
#               target = target.cuda()
        
#    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

def accuracy(model, data_loader, view_number):
        
    """Compute the train/test accuracy.

        Args:
            data_loader: Train/Test DataLoader.

        Returns:
            Train/Test accuracy in percentage.
    """
    model.train(False)
    model.eval()
    
    torch.no_grad()
    num_correct = 0
    num_total = 0
    Labels = torch.zeros(0,dtype=torch.int64)
    Preds = torch.zeros(0,dtype=torch.int64)
    if opt.cuda:
        Labels = Labels.cuda()
        Preds = Preds.cuda()
    for batch_idx, (input, labels) in enumerate(data_loader):
        # Data.
#        pdb.set_trace()
        if opt.cuda:
           labels = labels.cuda()
        
        Labels = torch.cat((Labels,labels),0)
        
        if view_number == 1:
               input1 = Variable(input[0])
               if opt.cuda:
                  input1 = input1.cuda()
               output = model(input1)
        if view_number == 2:
           input1, input2 = Variable(input[0]), Variable(input[1])
           if opt.cuda:
              input1 = input1.cuda()
              input2 = input2.cuda()
           output = model(input1,input2)
#               pdb.set_trace()
        if view_number == 3:
           input1, input2, input3 = Variable(input[0]), Variable(input[1]), Variable(input[2])
           if opt.cuda:
              input1 = input1.cuda()
              input2 = input2.cuda()
              input3 = input3.cuda()
           output = model(input1,input2,input3)
        if view_number == 4:
           input1, input2, input3, input4 = Variable(input[0]), Variable(input[1]), Variable(input[2]), Variable(input[3])
           if opt.cuda:
              input1 = input1.cuda()
              input2 = input2.cuda()
              input3 = input3.cuda()
              input4 = input4.cuda()             
           output = model(input1,input2,input3,input4)
        if view_number == 5:
           input1, input2, input3, input4, input5 = Variable(input[0]), Variable(input[1]), Variable(input[2]), Variable(input[3]), Variable(input[4])
           if opt.cuda:
              input1 = input1.cuda()
              input2 = input2.cuda()
              input3 = input3.cuda()
              input4 = input4.cuda()                  
              input4 = input5.cuda()
           output = model(input1,input2,input3,input4,input5)
#            X = X.to(device)
#            y = y.to(device)

        # Prediction.
#            pdb.set_trace()
        _, prediction = torch.max(output.data, 1)
        Preds = torch.cat((Preds,prediction),0)
            
        
        num_total += labels.size(0)
        num_correct += torch.sum(prediction == labels.data).item()
#        pdb.set_trace()
    
    confusion = []
    for i in range(0,len(Preds),50):
#        pdb.set_trace()
        rate =  (Preds[i:i+50]==Labels[i:i+50]).sum().cpu()
        rate = rate.to(dtype=torch.float)/50.0
        confusion.append(rate)
#            del X, y
    model.train(True)  # Set the model to training phase
#    pdb.set_trace()
    return 100 * num_correct / num_total, confusion


if __name__ == "__main__":
    main()
